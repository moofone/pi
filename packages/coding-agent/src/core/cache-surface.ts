import { createHash, randomUUID } from "node:crypto";
import { zstdDecompressSync } from "node:zlib";
import type { ModelsSimpleStreamOptions } from "@earendil-works/pi-ai";
import type { SessionManager } from "./session-manager.ts";

/** SSE only: final fetch arguments after payload/header hooks and lossless zstd decoding. */
export const CACHE_SURFACE_VERSION = "openai-codex-sse-request-prefix.v1";
export const CACHE_SURFACE_ENTRY = "pi.cache-surface.v1";
export type CacheSurfaceState = "unknown" | "changed" | "preserved";
export interface CacheSurfaceStatus {
	state: CacheSurfaceState;
	stage: "binding" | "request";
	requestId?: string;
	baselineRequestId?: string;
	fingerprintVersion: string;
	reasons: string[];
	differences: string[];
	/** Preservation is local request equality, never a promise of a provider cache hit. */
	outboundPrefixSha256?: string;
	cacheRead?: number;
	cacheNotice?: string;
}
export interface CacheSurfaceSemantic {
	tools: { name: string; definitionSha256: string }[];
	systemPromptSha256: string;
}
interface Snapshot {
	version: string;
	fields: Record<string, string>;
	input: string[];
	semantic: CacheSurfaceSemantic;
}
interface Receipt {
	v: 1;
	kind:
		| "prepared"
		| "submitted"
		| "ambiguous"
		| "cancelled"
		| "unsupported"
		| "observation"
		| "intent"
		| "notice"
		| "assistant";
	requestId: string;
	requestKind?: "regular" | "warm" | "diagnostic";
	snapshot?: Snapshot;
	baseRequestId?: string;
	suffix?: string[];
	status?: CacheSurfaceStatus;
	tools?: string[];
	intentSource?: "api" | "tools-option";
	assistantSha256?: string;
}
interface Baseline {
	requestId?: string;
	snapshot?: Snapshot;
	reason?: string;
}
const hash = (value: string) => createHash("sha256").update(value).digest("hex");
const isObject = (value: unknown): value is Record<string, unknown> =>
	typeof value === "object" && value !== null && !Array.isArray(value);
const strings = (value: unknown): value is string[] =>
	Array.isArray(value) && value.every((v) => typeof v === "string");

/** Semantic equality only; outbound hashes deliberately do NOT use this serializer. */
export function semanticHash(value: unknown): string {
	const canonical = (item: unknown): unknown => {
		if (Array.isArray(item)) return item.map(canonical);
		if (isObject(item))
			return Object.fromEntries(
				Object.keys(item)
					.sort()
					.map((key) => [key, canonical(item[key])]),
			);
		return item;
	};
	return hash(JSON.stringify(canonical(value)));
}
function validSnapshot(value: unknown): value is Snapshot {
	if (!isObject(value) || typeof value.version !== "string" || !isObject(value.fields) || !strings(value.input))
		return false;
	if (!Object.values(value.fields).every((v) => typeof v === "string") || !isObject(value.semantic)) return false;
	return (
		typeof value.semantic.systemPromptSha256 === "string" &&
		Array.isArray(value.semantic.tools) &&
		value.semantic.tools.every(
			(tool) => isObject(tool) && typeof tool.name === "string" && typeof tool.definitionSha256 === "string",
		)
	);
}
function toolDiff(a: CacheSurfaceSemantic, b: CacheSurfaceSemantic): string[] {
	const changes: string[] = [];
	for (const tool of b.tools) {
		const old = a.tools.find((t) => t.name === tool.name);
		if (!old) changes.push(`+${tool.name}`);
		else if (old.definitionSha256 !== tool.definitionSha256) changes.push(`definition:${tool.name}`);
	}
	for (const tool of a.tools) if (!b.tools.some((t) => t.name === tool.name)) changes.push(`-${tool.name}`);
	if (JSON.stringify(a.tools.map((t) => t.name)) !== JSON.stringify(b.tools.map((t) => t.name)))
		changes.push("tool-order");
	if (a.systemPromptSha256 !== b.systemPromptSha256) changes.push("system-prompt");
	return changes;
}

/** Branch-scoped observations and dispatch receipts. Contains hashes only, never prompts or auth. */
export class CacheSurfaceTracker {
	private firstReceipt = true;
	private bindingDifferences: string[] = [];
	private outcomeStatus?: CacheSurfaceStatus;
	private pendingAssistantRequestId?: string;
	private semantic: () => CacheSurfaceSemantic;
	private statusValue: CacheSurfaceStatus;
	private nextKind: "regular" | "diagnostic" = "regular";
	private readonly manager: Pick<SessionManager, "getBranch" | "appendCustomEntry">;
	private readonly publish: (status: CacheSurfaceStatus) => void;
	constructor(
		manager: Pick<SessionManager, "getBranch" | "appendCustomEntry">,
		semantic: () => CacheSurfaceSemantic,
		publish: (status: CacheSurfaceStatus) => void = () => {},
	) {
		this.manager = manager;
		this.publish = publish;
		this.semantic = semantic;
		this.statusValue = this.unknown("binding", "no-submitted-baseline");
	}
	get status(): CacheSurfaceStatus {
		return structuredClone(this.statusValue);
	}
	/** Explicit diagnostics cannot replace an unknown/changed historical baseline. */
	tagNextDiagnostic(): void {
		this.nextKind = "diagnostic";
	}
	private write(record: Receipt): void {
		this.manager.appendCustomEntry(CACHE_SURFACE_ENTRY, record);
	}
	private unknown(stage: "binding" | "request", reason: string, differences: string[] = []): CacheSurfaceStatus {
		return { state: "unknown", stage, fingerprintVersion: CACHE_SURFACE_VERSION, reasons: [reason], differences };
	}
	private emit(status: CacheSurfaceStatus): CacheSurfaceStatus {
		this.statusValue = structuredClone(status);
		this.publish(this.status);
		return status;
	}
	private baseline(): Baseline {
		const prepared = new Set<string>();
		let last: Baseline = { reason: "no-submitted-baseline" };
		const branch = this.manager.getBranch();
		const assistants = new Map<string, Set<string>>();
		for (const entry of branch) {
			if (
				entry.type === "custom" &&
				entry.customType === CACHE_SURFACE_ENTRY &&
				isObject(entry.data) &&
				entry.data.v === 1 &&
				entry.data.kind === "assistant" &&
				typeof entry.data.requestId === "string" &&
				typeof entry.data.assistantSha256 === "string"
			) {
				const ids = assistants.get(entry.data.assistantSha256) ?? new Set<string>();
				ids.add(entry.data.requestId);
				assistants.set(entry.data.assistantSha256, ids);
			}
		}
		for (const entry of branch) {
			if (entry.type === "message" && entry.message.role === "assistant" && last.snapshot) {
				const ids = assistants.get(hash(JSON.stringify(entry.message)));
				if (!last.requestId || !ids?.delete(last.requestId)) last = { reason: "unmatched-assistant-entry" };
			}
			if (entry.type !== "custom" || entry.customType !== CACHE_SURFACE_ENTRY) continue;
			const data = entry.data;
			if (!isObject(data) || data.v !== 1 || typeof data.kind !== "string" || typeof data.requestId !== "string") {
				last = { reason: "unreadable-record" };
				continue;
			}
			if (
				data.kind === "observation" ||
				data.kind === "intent" ||
				data.kind === "notice" ||
				data.kind === "assistant"
			)
				continue;
			if (data.kind === "prepared") {
				prepared.add(data.requestId);
				continue;
			}
			if (data.kind === "cancelled") {
				prepared.delete(data.requestId);
				continue;
			}
			if (data.kind !== "submitted") {
				prepared.delete(data.requestId);
				last = { reason: "ambiguous-dispatch" };
				continue;
			}
			if (!prepared.delete(data.requestId)) {
				last = { reason: "unreadable-record" };
				continue;
			}
			let snapshot: Snapshot | undefined;
			if (validSnapshot(data.snapshot)) snapshot = { ...data.snapshot, input: [...data.snapshot.input] };
			else if (typeof data.baseRequestId === "string" && strings(data.suffix)) {
				// Append to our own replay buffer, never to persisted record data. Retaining
				// every expanded prefix would make a long session quadratic in memory.
				if (last.requestId === data.baseRequestId && last.snapshot && prepared.size === 0) {
					snapshot = last.snapshot;
					snapshot.input.push(...data.suffix);
				}
			}
			if (!snapshot) {
				last = { reason: "unreadable-record" };
				continue;
			}
			if (snapshot.version !== CACHE_SURFACE_VERSION) {
				last = { reason: "incompatible-fingerprint-version" };
				continue;
			}
			prepared.clear(); // A newer confirmed full capture can establish a new baseline.
			last = { requestId: data.requestId, snapshot };
		}
		if (prepared.size) return { reason: "ambiguous-dispatch" };
		return last;
	}
	observe(): CacheSurfaceStatus {
		const baseline = this.baseline();
		const differences = baseline.snapshot ? toolDiff(baseline.snapshot.semantic, this.semantic()) : this.legacyDiff();
		const status: CacheSurfaceStatus = {
			...this.unknown("binding", baseline.reason ?? "pending-final-check", differences),
			baselineRequestId: baseline.requestId,
			state: baseline.snapshot && differences.length ? "changed" : "unknown",
		};
		this.bindingDifferences = differences;
		this.write({ v: 1, kind: "observation", requestId: randomUUID(), status });
		return this.emit(status);
	}
	private legacyDiff(): string[] {
		const tools = new Map<string, { name: string; definitionSha256: string }>();
		let found = false;
		for (const entry of this.manager.getBranch()) {
			if (entry.type !== "message" || entry.message.role !== "system") continue;
			for (const tool of entry.message.toolsAdded ?? []) {
				found = true;
				tools.set(tool.name, {
					name: tool.name,
					definitionSha256: semanticHash({
						name: tool.name,
						description: tool.description,
						parameters: tool.parameters,
					}),
				});
			}
			for (const tool of entry.message.toolsRemoved ?? []) tools.delete(tool.name);
		}
		return found
			? toolDiff(
					{ tools: [...tools.values()], systemPromptSha256: this.semantic().systemPromptSha256 },
					this.semantic(),
				)
			: [];
	}
	/** Caller must establish explicit session intent; host availability must never call this. */
	recordToolIntent(tools: string[], intentSource: "api" | "tools-option" = "api"): void {
		this.write({ v: 1, kind: "intent", requestId: randomUUID(), tools: [...tools], intentSource });
	}
	getSubmittedToolNames(): string[] | undefined {
		return this.baseline().snapshot?.semantic.tools.map((tool) => tool.name);
	}
	getToolIntent(): string[] | undefined {
		let tools: string[] | undefined;
		for (const entry of this.manager.getBranch()) {
			if (
				entry.type === "message" &&
				entry.message.role === "system" &&
				(entry.message.toolsAdded?.length || entry.message.toolsRemoved?.length)
			)
				tools = undefined;
			if (entry.type !== "custom" || entry.customType !== CACHE_SURFACE_ENTRY || !isObject(entry.data)) continue;
			if (entry.data.v === 1 && entry.data.kind === "intent" && strings(entry.data.tools))
				tools = [...entry.data.tools];
			// A submitted request is evidence, not a later explicit intent override.
		}
		return tools;
	}
	/** Prevent a custom adapter that bypasses final capture from inheriting a prior verdict. */
	beginRequest(): void {
		this.pendingAssistantRequestId = undefined;
		this.outcomeStatus = this.emit(this.unknown("request", "pending-final-check"));
	}
	unsupported(): void {
		const requestId = randomUUID();
		this.pendingAssistantRequestId = requestId;
		this.write({ v: 1, kind: "unsupported", requestId });
		this.outcomeStatus = this.emit({ ...this.unknown("request", "final-payload-unavailable"), requestId });
	}
	/** No exclusions from body. Authorization is replaced by final authenticated account identity;
	 * compression/length are lossless transport framing. Every other header participates. */
	private capture(input: string | URL | Request, init: RequestInit | undefined): Snapshot | undefined {
		if (String(input) !== "https://chatgpt.com/backend-api/codex/responses" || !init || init.method !== "POST")
			return undefined;
		const headers = new Headers(init.headers);
		if (!headers.get("chatgpt-account-id") || !headers.get("session-id")) return undefined;
		let body: string;
		try {
			if (typeof init.body === "string") body = init.body;
			else if (init.body instanceof Uint8Array && headers.get("content-encoding") === "zstd")
				body = zstdDecompressSync(init.body).toString("utf8");
			else return undefined;
			const payload: unknown = JSON.parse(body);
			if (
				!isObject(payload) ||
				!Array.isArray(payload.input) ||
				body !== JSON.stringify(payload) ||
				"previous_response_id" in payload
			)
				return undefined;
			const fields: Record<string, string> = {};
			fields["body-field-order"] = hash(JSON.stringify(Object.keys(payload)));
			for (const [key, value] of Object.entries(payload))
				if (key !== "input") fields[`body:${key}`] = hash(JSON.stringify(value));
			for (const [key, value] of headers) {
				if (["authorization", "content-encoding", "content-length"].includes(key)) continue;
				fields[`header:${key}`] = hash(value);
			}
			return {
				version: CACHE_SURFACE_VERSION,
				fields,
				input: payload.input.map((item) => hash(JSON.stringify(item))),
				semantic: this.semantic(),
			};
		} catch {
			return undefined;
		}
	}
	private compare(baseline: Baseline, snapshot: Snapshot | undefined, complete: boolean): CacheSurfaceStatus {
		if (!baseline.snapshot || !baseline.requestId)
			return this.unknown("request", baseline.reason ?? "no-submitted-baseline", this.bindingDifferences);
		const differences = snapshot ? toolDiff(baseline.snapshot.semantic, snapshot.semantic) : [];
		if (snapshot) {
			for (const key of new Set([...Object.keys(baseline.snapshot.fields), ...Object.keys(snapshot.fields)])) {
				if (baseline.snapshot.fields[key] !== snapshot.fields[key]) differences.push(key);
			}
			if (
				baseline.snapshot.input.length > snapshot.input.length ||
				baseline.snapshot.input.some((h, i) => snapshot.input[i] !== h)
			)
				differences.push("retained-message-prefix");
		}
		return {
			state: differences.length ? "changed" : complete && snapshot ? "preserved" : "unknown",
			stage: "request",
			fingerprintVersion: CACHE_SURFACE_VERSION,
			baselineRequestId: baseline.requestId,
			differences,
			reasons: complete && snapshot ? [] : ["unchecked-transformation"],
			outboundPrefixSha256: snapshot
				? hash(
						JSON.stringify({
							fields: snapshot.fields,
							input: snapshot.input.slice(0, baseline.snapshot.input.length),
						}),
					)
				: undefined,
		};
	}
	/** An assistant without a matching request receipt invalidates baseline selection. */
	recordAssistant(message: unknown): void {
		if (!this.pendingAssistantRequestId) return;
		this.write({
			v: 1,
			kind: "assistant",
			requestId: this.pendingAssistantRequestId,
			assistantSha256: hash(JSON.stringify(message)),
		});
		this.pendingAssistantRequestId = undefined;
	}
	/** Cache outcome is associated with the saved pre-dispatch comparison, not the new baseline. */
	recordOutcome(event: unknown): void {
		if (
			!isObject(event) ||
			event.type !== "response.completed" ||
			!isObject(event.response) ||
			!isObject(event.response.usage)
		)
			return;
		const details = event.response.usage.input_tokens_details;
		if (!isObject(details) || typeof details.cached_tokens !== "number" || !this.outcomeStatus) return;
		const original = this.outcomeStatus;
		const cacheRead = details.cached_tokens;
		const cacheNotice =
			cacheRead === 0 && original.state === "preserved"
				? "Outbound surface preserved within recorded scope; provider reported no cache reuse."
				: cacheRead === 0
					? `Provider reported no cache reuse. ${formatCacheSurfaceNotice(original)}`
					: `Provider cache read: ${cacheRead} tokens. ${formatCacheSurfaceNotice(original)}`;
		const status = { ...original, cacheRead, cacheNotice };
		this.write({ v: 1, kind: "notice", requestId: original.requestId ?? randomUUID(), status });
		this.emit(status);
		this.outcomeStatus = undefined;
	}
	wrapFetch(
		options: ModelsSimpleStreamOptions,
		kind: "regular" | "warm" | "diagnostic" = this.nextKind,
	): typeof fetch {
		this.nextKind = "regular";
		const baseFetch = options.fetch ?? globalThis.fetch;
		const completeTransport = options.transport === "sse" && !options.fetch;
		return async (input, init) => {
			const baseline = this.baseline();
			const snapshot = this.capture(input, init);
			const complete = completeTransport && !!snapshot;
			const requestId = randomUUID();
			const status = { ...this.compare(baseline, snapshot, complete), requestId };
			this.emit(status); // Warnings complete before transport invocation.
			if (kind !== "warm") {
				this.outcomeStatus = structuredClone(status);
				this.pendingAssistantRequestId = requestId;
			}
			if (kind === "warm" && status.state !== "preserved")
				throw new Error("Cache warm blocked: surface is not preserved");
			if (init?.signal?.aborted) throw init.signal.reason ?? new Error("Request aborted before handoff");
			this.write({ v: 1, kind: "prepared", requestId, requestKind: kind, status });
			let response: Response;
			try {
				response = await baseFetch(input, init);
			} catch (error) {
				this.write({ v: 1, kind: "ambiguous", requestId, status });
				throw error;
			}
			if (!complete || !snapshot || (kind === "diagnostic" && status.state !== "preserved")) {
				this.write({ v: 1, kind: "ambiguous", requestId, status });
				return response;
			}
			// A Response confirms handoff, even if the provider rejected the request. Never infer cache residence.
			const compact = !this.firstReceipt && baseline.snapshot && status.state === "preserved";
			this.write({
				v: 1,
				kind: "submitted",
				requestId,
				requestKind: kind,
				status,
				...(compact
					? { baseRequestId: baseline.requestId, suffix: snapshot.input.slice(baseline.snapshot?.input.length) }
					: { snapshot }),
			});
			this.firstReceipt = false;
			return response;
		};
	}
}

export function formatCacheSurfaceNotice(status: CacheSurfaceStatus): string {
	if (status.state === "changed")
		return `Cache surface changed locally: ${status.differences.join(", ")}. The next request may cold-start and rebill context.${status.reasons.length ? ` Coverage: ${status.reasons.join(", ")}.` : ""}`;
	if (status.state === "unknown")
		return `Cache surface unknown: ${status.reasons.join(", ")}${status.differences.length ? ` (${status.differences.join(", ")})` : ""}.`;
	return "Outbound surface preserved within recorded scope; provider cache reuse is not guaranteed.";
}
