import type { AgentMessage, StreamFn } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	type Model,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { generateSummaryWithUsage } from "../src/core/compaction/index.ts";

/**
 * Regression tests for compacting a session that was built under a
 * large-context model after switching to a model with a much smaller
 * context window (e.g. 1M → swe-2). The serialized conversation can far
 * exceed the new model's window, so summarization must chunk and merge
 * instead of failing with "prompt is too long".
 */

const SMALL_MODEL: Model<"anthropic-messages"> = {
	id: "small-model",
	name: "Small Model",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 40000,
	maxTokens: 8192,
};

const BIG_MODEL: Model<"anthropic-messages"> = { ...SMALL_MODEL, contextWindow: 1000000 };

function response(text: string): AssistantMessage {
	return {
		...fauxAssistantMessage(""),
		content: [{ type: "text", text }],
		api: SMALL_MODEL.api,
		provider: SMALL_MODEL.provider,
		model: SMALL_MODEL.id,
	};
}

/** StreamFn that echoes a deterministic summary and records every context. */
function recordingStreamFn(contexts: TranscriptContext[], reply = "partial summary"): StreamFn {
	return (_model, context) => {
		contexts.push(context);
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() => stream.push({ type: "done", reason: "stop", message: response(reply) }));
		return stream;
	};
}

function promptTextOf(context: TranscriptContext): string {
	const parts: string[] = [];
	for (const msg of context.messages) {
		if (msg.role !== "user") continue;
		if (typeof msg.content === "string") {
			parts.push(msg.content);
		} else if (Array.isArray(msg.content)) {
			parts.push(
				...msg.content.filter((b): b is { type: "text"; text: string } => b.type === "text").map((b) => b.text),
			);
		}
	}
	return parts.join("");
}

function totalPromptChars(context: TranscriptContext): number {
	let chars = promptTextOf(context).length;
	for (const msg of context.messages) {
		if (msg.role !== "system") continue;
		chars += typeof msg.content === "string" ? msg.content.length : 0;
	}
	return chars;
}

function makeMessages(count: number, charsEach: number): AgentMessage[] {
	const messages: AgentMessage[] = [];
	for (let i = 0; i < count; i++) {
		messages.push({
			role: "user",
			content: `message ${i}: ${"x".repeat(charsEach)}`,
			timestamp: Date.now(),
		});
	}
	return messages;
}

/** Assert every recorded request fits the small model's context window. */
function expectAllRequestsFit(contexts: TranscriptContext[], model: Model<any>, maxTokens: number) {
	expect(contexts.length).toBeGreaterThan(0);
	for (const context of contexts) {
		const promptTokens = Math.ceil(totalPromptChars(context) / 4);
		expect(promptTokens + maxTokens).toBeLessThanOrEqual(model.contextWindow);
	}
}

describe("summarization with a shrunk context window", () => {
	it("sends a single request when the conversation fits the window", async () => {
		const contexts: TranscriptContext[] = [];
		const result = await generateSummaryWithUsage(
			makeMessages(4, 100),
			SMALL_MODEL,
			16384,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			recordingStreamFn(contexts),
		);
		expect(result.text).toBe("partial summary");
		expect(contexts).toHaveLength(1);
	});

	it("chunks and merges a conversation larger than the model window", async () => {
		const contexts: TranscriptContext[] = [];
		// ~20 * 20KB = 400KB serialized >> 40k-token window.
		const result = await generateSummaryWithUsage(
			makeMessages(20, 20000),
			SMALL_MODEL,
			16384,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			recordingStreamFn(contexts),
		);
		expect(result.text).toBe("partial summary");
		// More than one summarization call must have happened (chunks + merges).
		expect(contexts.length).toBeGreaterThan(2);
		expectAllRequestsFit(contexts, SMALL_MODEL, 8192);
		// Some requests must carry the chunk prompt, others the merge prompt.
		const prompts = contexts.map(promptTextOf);
		expect(prompts.some((p) => p.includes("one part of a long conversation"))).toBe(true);
		expect(prompts.some((p) => p.includes("partial summaries of consecutive parts"))).toBe(true);
	});

	it("folds the previous summary into a final update request", async () => {
		const contexts: TranscriptContext[] = [];
		const result = await generateSummaryWithUsage(
			makeMessages(20, 20000),
			SMALL_MODEL,
			16384,
			undefined,
			undefined,
			undefined,
			undefined,
			"PREVIOUS SUMMARY CONTENT",
			undefined,
			recordingStreamFn(contexts),
		);
		expect(result.text).toBe("partial summary");
		const prompts = contexts.map(promptTextOf);
		expect(prompts.some((p) => p.includes("<previous-summary>"))).toBe(true);
		// Chunk requests must not repeat the previous summary.
		const chunkPrompts = prompts.filter((p) => p.includes("one part of a long conversation"));
		expect(chunkPrompts.length).toBeGreaterThan(1);
		expect(chunkPrompts.every((p) => !p.includes("PREVIOUS SUMMARY CONTENT"))).toBe(true);
		expectAllRequestsFit(contexts, SMALL_MODEL, 8192);
	});

	it("truncates a single message that alone exceeds the window", async () => {
		const contexts: TranscriptContext[] = [];
		const result = await generateSummaryWithUsage(
			makeMessages(1, 400000), // ~100k tokens in one message, window is 40k
			SMALL_MODEL,
			16384,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			recordingStreamFn(contexts),
		);
		expect(result.text).toBe("partial summary");
		expectAllRequestsFit(contexts, SMALL_MODEL, 8192);
		expect(promptTextOf(contexts[0])).toContain("truncated to fit the summarization model");
	});

	it("keeps the single-request path for large-window models", async () => {
		const contexts: TranscriptContext[] = [];
		await generateSummaryWithUsage(
			makeMessages(20, 20000),
			BIG_MODEL,
			16384,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			undefined,
			recordingStreamFn(contexts),
		);
		expect(contexts).toHaveLength(1);
	});
});
