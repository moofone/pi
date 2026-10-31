# Cache-surface evidence

`createAgentSession()` returns `cacheSurface` and accepts an optional `onCacheSurface(status)` callback. Both TUI and embedding hosts use the same SDK tracker; hosts should forward its status rather than recompute a fingerprint.

## Status

- `unknown`: no trustworthy submitted baseline, pending final check, unsupported transport/adapter, ambiguous handoff or incomplete coverage.
- `changed`: a known tool/order/prompt/configuration/affinity/retained-prefix/serialization difference. Other gaps remain in `reasons`.
- `preserved`: the final serialized logical request matches a trustworthy branch baseline within the recorded capture scope. **Not a provider cache-hit guarantee.**

Binding only writes an observation. In particular, binding-time semantic equality is `unknown/pending-final-check`. The final request comparison includes `requestId` and `baselineRequestId`; cache outcomes retain that original comparison, not a later baseline. `cacheRead` is reported provider usage, separate from local equality. Non-blocking notices distinguish a changed surface, unknown coverage and a preserved request with no provider reuse.

## Supported boundary

Version `openai-codex-sse-request-prefix.v1`: native Codex Responses SSE, default native fetch and the canonical Codex endpoint. Capture is after hooks, conversion, final headers and JSON serialization. Lossless zstd is decoded.

Exact emitted JSON values/order, retained input-item prefix hashes and final headers participate. Authorization token bytes are excluded in favor of the hashed authenticated account identity; compression/content-length framing is excluded. SDK records contain hashes/diagnostics, not prompts or credentials.

WebSocket continuation/delta requests, custom per-request fetch, other adapters, noncanonical endpoints and incomplete transformations remain unknown. Do not substitute a pre-provider hook or sorted semantic hash for final capture.

## Persistence and intent

Branch custom entries use `pi.cache-surface.v1`: observations, explicit tool intent, prepared requests, submitted/ambiguous outcomes, assistant correlation and notices. Confirmed transport handoff establishes a submitted request, not provider acceptance/cache residence. Unsupported or unmatched assistant history invalidates preservation. A later complete confirmed regular request can establish a fresh baseline; warm/diagnostic traffic cannot launder an unknown/changed one.

First submission after resume writes a full checkpoint. Preserved subsequent submissions use a prior-prefix reference plus suffix hashes. Closing without sending never establishes or replaces a baseline.

Use `session.setActiveToolsByName()` (or the host's session-scoped tool API) for explicit intent. Unsent changes survive reopen. Existing transcript loadout restoration is preserved; submitted evidence prevents previously unmentioned automatic registrations from silently extending it. Missing tools remain visible as drift, never falsely preserved. Forks/siblings without resolvable branch-local request/assistant evidence remain unknown.

Automatic warming refuses a non-preserved surface. Embedders with a separate warming controller must apply the same refusal; evidence stability does not prove provider cache residence or TTL.
