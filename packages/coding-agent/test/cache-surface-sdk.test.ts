import { copyFileSync, mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { CacheSurfaceStatus } from "../src/core/cache-surface.ts";
import type { ExtensionFactory } from "../src/core/extensions/types.ts";
import { ModelRuntime } from "../src/core/model-runtime.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";

const model: Model<"openai-codex-responses"> = {
	id: "synthetic",
	name: "Synthetic",
	provider: "openai-codex",
	api: "openai-codex-responses",
	baseUrl: "https://chatgpt.com/backend-api",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 4096,
};
const token = `e30.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "mock-account" } })).toString("base64url")}.signature`;

describe("SDK final Codex SSE surface boundary (no network)", () => {
	let root: string;
	let agentDir: string;
	let runtime: ModelRuntime;
	let extraTool: boolean;
	let lateChange: boolean;
	let headerChange: boolean;
	let sessions: AgentSession[];
	let statuses: CacheSurfaceStatus[];
	const factory: ExtensionFactory = (pi) => {
		for (const name of extraTool ? ["session_tool", "host_only"] : ["session_tool"]) {
			pi.registerTool({
				name,
				label: name,
				description: name,
				parameters: Type.Object({}),
				execute: async () => ({ content: [], details: {} }),
			});
		}
		pi.on("before_provider_request", (event) => {
			if (!lateChange || !event.payload || typeof event.payload !== "object") return;
			return { ...event.payload, instructions: "Deliberately different early instructions" };
		});
		pi.on("before_provider_headers", (event) => {
			if (headerChange) event.headers["x-routing-test"] = "changed";
		});
	};
	beforeEach(async () => {
		root = mkdtempSync(join(tmpdir(), "pi-cache-surface-sdk-"));
		agentDir = join(root, "agent");
		mkdirSync(agentDir);
		extraTool = false;
		lateChange = false;
		headerChange = false;
		sessions = [];
		statuses = [];
		const auth = AuthStorage.create(join(agentDir, "auth.json"));
		await auth.modify("openai-codex", async () => ({
			type: "oauth",
			access: token,
			refresh: "mock",
			expires: Date.now() + 3600000,
		}));
		runtime = await ModelRuntime.create({
			authPath: join(agentDir, "auth.json"),
			modelsPath: join(agentDir, "models.json"),
		});
		vi.stubGlobal(
			"fetch",
			vi.fn(async () => {
				// Actual native adapter conversion and final headers, but synthetic response only.
				const event = {
					type: "response.completed",
					response: {
						id: "response_mock",
						status: "completed",
						output: [],
						model: model.id,
						usage: {
							input_tokens: 1000,
							output_tokens: 1,
							total_tokens: 1001,
							input_tokens_details: { cached_tokens: 0 },
							output_tokens_details: { reasoning_tokens: 0 },
						},
					},
				};
				return new Response(`data: ${JSON.stringify(event)}\n\n`, {
					headers: { "content-type": "text/event-stream" },
				});
			}),
		);
	});
	afterEach(() => {
		for (const session of sessions) session.dispose();
		vi.unstubAllGlobals();
		rmSync(root, { recursive: true, force: true });
	});
	async function make(manager = SessionManager.create(root, join(root, "sessions")), tools?: string[]) {
		const settingsManager = SettingsManager.inMemory({
			transport: "sse",
			cacheWarming: "off",
			packages: [],
			extensions: [],
			skills: [],
		});
		const loader = new DefaultResourceLoader({
			cwd: root,
			agentDir,
			settingsManager,
			extensionFactories: [factory],
			noSkills: true,
		});
		await loader.reload();
		const result = await createAgentSession({
			cwd: root,
			agentDir,
			sessionManager: manager,
			model,
			modelRuntime: runtime,
			thinkingLevel: "off",
			settingsManager,
			resourceLoader: loader,
			tools,
			onCacheSurface: (status) => statuses.push(status),
		});
		sessions.push(result.session);
		await result.session.bindExtensions({});
		return { ...result, manager };
	}
	it("captures real converted/compressed native requests and correlates persisted assistants", async () => {
		const first = await make();
		await first.session.prompt("Seed");
		expect(first.cacheSurface.status.state).toBe("unknown");
		await first.session.prompt("Control");
		expect(
			first.cacheSurface.status,
			JSON.stringify({
				statuses,
				receipts: first.manager.getBranch().filter((entry) => entry.type === "custom"),
				urls: vi.mocked(globalThis.fetch).mock.calls.map(([url]) => String(url)),
			}),
		).toMatchObject({ state: "preserved", reasons: [] });
		expect(first.cacheSurface.status.cacheNotice).toContain("provider reported no cache reuse");
		const file = first.manager.getSessionFile();
		if (!file) throw new Error("missing session file");
		first.session.dispose();
		const resumed = await make(SessionManager.open(file));
		expect(resumed.cacheSurface.status.state).toBe("unknown");
		expect(resumed.cacheSurface.status.reasons).toContain("pending-final-check");
		await resumed.session.prompt("Resumed");
		expect(resumed.cacheSurface.status.state).toBe("preserved");
	});
	it("suppresses unmentioned automatic host tools, while explicit modifiers remain intent", async () => {
		const first = await make();
		await first.session.prompt("Seed");
		const file = first.manager.getSessionFile();
		if (!file) throw new Error("missing session");
		first.session.dispose();
		extraTool = true;
		const resumed = await make(SessionManager.open(file));
		expect(resumed.session.getActiveToolNames()).not.toContain("host_only");
		await resumed.session.prompt("Resume");
		expect(resumed.cacheSurface.status.state).toBe("preserved");
		resumed.session.dispose();
		const explicit = await make(SessionManager.open(file), ["+host_only"]);
		expect(explicit.session.getActiveToolNames()).toContain("host_only");
		expect(explicit.cacheSurface.status.differences).toContain("+host_only");
		await explicit.session.prompt("Intentional tools");
		expect(explicit.cacheSurface.status.state).toBe("changed");
	});
	it("detects late payload and header hooks before fetch, not just binding equality", async () => {
		const first = await make();
		await first.session.prompt("Seed");
		lateChange = true;
		headerChange = true;
		await first.session.prompt("Changed");
		expect(first.cacheSurface.status.state).toBe("changed");
		expect(first.cacheSurface.status.differences).toContain("body:instructions");
		expect(first.cacheSurface.status.differences).toContain("header:x-routing-test");
		expect(statuses.some((s) => s.stage === "request" && s.state === "changed" && s.cacheRead === undefined)).toBe(
			true,
		);
	});
	it("keeps both copied branch requests comparable and does not use a sibling record", async () => {
		const first = await make();
		await first.session.prompt("Seed");
		const file = first.manager.getSessionFile();
		if (!file) throw new Error("missing session");
		first.session.dispose();
		const copy = join(root, "copy.jsonl");
		copyFileSync(file, copy);
		const a = await make(SessionManager.open(file));
		const b = await make(SessionManager.open(copy));
		await a.session.prompt("Identical next prompt");
		await b.session.prompt("Identical next prompt");
		expect(a.cacheSurface.status.state).toBe("preserved");
		expect(b.cacheSurface.status.outboundPrefixSha256).toBe(a.cacheSurface.status.outboundPrefixSha256);
	});
});
