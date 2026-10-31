import { zstdCompressSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CacheSurfaceSemantic, CacheSurfaceStatus } from "../src/core/cache-surface.ts";
import { CACHE_SURFACE_ENTRY, CacheSurfaceTracker, semanticHash } from "../src/core/cache-surface.ts";
import { SessionManager } from "../src/core/session-manager.ts";

const endpoint = "https://chatgpt.com/backend-api/codex/responses";
const headers = {
	"chatgpt-account-id": "synthetic-account",
	"session-id": "same-session",
	authorization: "Bearer SECRET",
};
const semantic = (): CacheSurfaceSemantic => ({
	tools: [{ name: "read", definitionSha256: semanticHash({ name: "read", description: "read", parameters: {} }) }],
	systemPromptSha256: semanticHash("system"),
});
const payload = (input: unknown[] = [{ role: "user", content: "large retained prefix" }]) => ({
	model: "synthetic",
	instructions: "system",
	input,
	prompt_cache_key: "same-session",
	reasoning: { effort: "none" },
});
const init = (body = payload(), extra: RequestInit = {}): RequestInit => ({
	method: "POST",
	headers,
	body: JSON.stringify(body),
	...extra,
});
const track = (manager = SessionManager.inMemory(), surface = semantic, notices: CacheSurfaceStatus[] = []) => ({
	manager,
	tracker: new CacheSurfaceTracker(manager, surface, (status) => notices.push(status)),
	notices,
});
const send = (tracker: CacheSurfaceTracker, request = init()) =>
	tracker.wrapFetch({ transport: "sse" })(endpoint, request);
const completed = (cached: number) => ({
	type: "response.completed",
	response: { usage: { input_tokens_details: { cached_tokens: cached } } },
});

afterEach(() => vi.unstubAllGlobals());
describe("submitted cache surface evidence", () => {
	it("semantic canonicalization does not imply emitted ordering equality", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		expect(semanticHash({ a: 1, b: 2 })).toBe(semanticHash({ b: 2, a: 1 }));
		const { tracker } = track();
		await send(tracker, init(payload([{ a: 1, b: 2 }])));
		await send(tracker, init(payload([{ b: 2, a: 1 }])));
		expect(tracker.status.state).toBe("changed");
		expect(tracker.status.differences).toContain("retained-message-prefix");
	});
	it("observes without baselining; first send remains unknown; normal suffix preserves", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker, manager } = track();
		expect(tracker.observe().state).toBe("unknown");
		expect(new CacheSurfaceTracker(manager, semantic).observe().reasons).toContain("no-submitted-baseline");
		await send(tracker);
		expect(tracker.status.state).toBe("unknown");
		const restored = new CacheSurfaceTracker(manager, semantic);
		expect(restored.observe().reasons).toContain("pending-final-check");
		await send(restored, init(payload([...payload().input, { role: "assistant", content: "OK" }])));
		expect(restored.status.state).toBe("preserved");
	});
	it.each(["instructions", "model", "prompt_cache_key", "reasoning"])(
		"checks late %s mutation before invoking transport",
		async (field) => {
			const fetch = vi.fn(async () => new Response(""));
			vi.stubGlobal("fetch", fetch);
			const { tracker, notices } = track();
			await send(tracker);
			fetch.mockImplementationOnce(async () => {
				expect(notices.at(-1)?.state).toBe("changed");
				expect(notices.at(-1)?.differences).toContain(`body:${field}`);
				return new Response("");
			});
			await send(tracker, init({ ...payload(), [field]: "changed" }));
		},
	);
	it("checks authenticated identity and session headers without persisting credentials", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker, manager } = track();
		await send(tracker);
		await send(tracker, init(payload(), { headers: { ...headers, "chatgpt-account-id": "other" } }));
		expect(tracker.status.differences).toContain("header:chatgpt-account-id");
		expect(JSON.stringify(manager.getBranch())).not.toContain("SECRET");
		expect(JSON.stringify(manager.getBranch())).not.toContain("synthetic-account");
	});
	it("normal unchanged receipts are compact and reloadable; first after resume is full", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker, manager } = track();
		await send(tracker);
		await send(tracker, init(payload([...payload().input, { role: "assistant", content: "OK" }])));
		const last = manager.getBranch().at(-1);
		expect(last?.type).toBe("custom");
		if (last?.type === "custom")
			expect(last.data).toMatchObject({
				kind: "submitted",
				suffix: expect.any(Array),
				baseRequestId: expect.any(String),
			});
		const resumed = new CacheSurfaceTracker(manager, semantic);
		await send(
			resumed,
			init(payload([...payload().input, { role: "assistant", content: "OK" }, { role: "user", content: "next" }])),
		);
		expect(resumed.status.state).toBe("preserved");
		const receipt = manager.getBranch().at(-1);
		if (receipt?.type === "custom") expect(receipt.data).toHaveProperty("snapshot");
	});
	it("drift survives observe/close/reopen without replacing submitted baseline", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker, manager } = track();
		await send(tracker);
		const changed = () => ({
			...semantic(),
			tools: [...semantic().tools, { name: "host_only", definitionSha256: "different" }],
		});
		expect(new CacheSurfaceTracker(manager, changed).observe().differences).toContain("+host_only");
		const again = new CacheSurfaceTracker(manager, changed);
		expect(again.observe().state).toBe("changed");
		expect(again.status.differences).toContain("+host_only");
	});
	it("cancellation before handoff retains baseline, ambiguous fetch cannot fall back", async () => {
		const fetch = vi.fn(async () => new Response(""));
		vi.stubGlobal("fetch", fetch);
		const { tracker, manager } = track();
		await send(tracker);
		const controller = new AbortController();
		controller.abort();
		await expect(send(tracker, init(payload(), { signal: controller.signal }))).rejects.toBeDefined();
		expect(fetch).toHaveBeenCalledTimes(1);
		expect(new CacheSurfaceTracker(manager, semantic).observe().reasons).toContain("pending-final-check");
		fetch.mockRejectedValueOnce(new Error("ambiguous network failure"));
		await expect(send(tracker)).rejects.toThrow("ambiguous");
		expect(new CacheSurfaceTracker(manager, semantic).observe().reasons).toContain("ambiguous-dispatch");
		await send(tracker);
		expect(tracker.status.state).toBe("unknown");
		expect(new CacheSurfaceTracker(manager, semantic).observe().reasons).toContain("pending-final-check");
	});
	it("branch selection never uses sibling-only submitted records", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker, manager } = track();
		await send(tracker);
		const branch = manager.getLeafId();
		if (!branch) throw new Error("missing baseline");
		await send(tracker, init({ ...payload(), instructions: "sibling" }));
		manager.branch(branch);
		await send(tracker);
		expect(tracker.status.state).toBe("preserved");
	});
	it("warm/diagnostic cannot launder unknown or changed surface", async () => {
		const fetch = vi.fn(async () => new Response(""));
		vi.stubGlobal("fetch", fetch);
		const { tracker, manager } = track();
		await expect(tracker.wrapFetch({ transport: "sse" }, "warm")(endpoint, init())).rejects.toThrow("not preserved");
		expect(fetch).not.toHaveBeenCalled();
		await tracker.wrapFetch({ transport: "sse" }, "diagnostic")(endpoint, init());
		expect(new CacheSurfaceTracker(manager, semantic).observe().state).toBe("unknown");
		expect(new CacheSurfaceTracker(manager, semantic).getSubmittedToolNames()).toBeUndefined();
	});
	it("custom fetch, auto/WebSocket transport, and malformed capture remain unknown", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker } = track();
		await send(tracker);
		await tracker.wrapFetch({ transport: "auto" })(endpoint, init());
		expect(tracker.status.state).not.toBe("preserved");
		await tracker.wrapFetch({ transport: "sse", fetch: async () => new Response("") })(endpoint, init());
		expect(tracker.status.state).not.toBe("preserved");
	});
	it("supports lossless zstd at actual final SSE boundary", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker } = track();
		await send(tracker);
		await send(
			tracker,
			init(payload(), {
				headers: { ...headers, "content-encoding": "zstd" },
				body: new Uint8Array(zstdCompressSync(JSON.stringify(payload()))),
			}),
		);
		expect(tracker.status.state).toBe("preserved");
	});
	it("rejects unreadable/version-incompatible records without stale fallback", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker, manager } = track();
		await send(tracker);
		manager.appendCustomEntry(CACHE_SURFACE_ENTRY, { v: 99 });
		expect(new CacheSurfaceTracker(manager, semantic).observe().reasons).toContain("unreadable-record");
	});
	it("notices retain original request comparison after new baseline is persisted", async () => {
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => new Response("")),
		);
		const { tracker } = track();
		await send(tracker);
		await send(tracker, init({ ...payload(), instructions: "changed" }));
		const requestId = tracker.status.requestId;
		tracker.recordOutcome(completed(0));
		expect(tracker.status.state).toBe("changed");
		expect(tracker.status.requestId).toBe(requestId);
		expect(tracker.status.cacheNotice).toContain("changed locally");
		await send(tracker, init({ ...payload(), instructions: "changed" }));
		tracker.recordOutcome(completed(0));
		expect(tracker.status.state).toBe("preserved");
		expect(tracker.status.cacheNotice).toContain("provider reported no cache reuse");
	});
});
