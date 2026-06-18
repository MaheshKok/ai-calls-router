# ADR-0001: Canonical Contract And Codex Bypass

## Status

Accepted.

## Context

`ai-calls-router` uses Anthropic Messages as its internal canonical request and response contract. Claude Code already speaks that wire, while Hermes Chat Completions and OpenAI Responses clients use adapters at the proxy edge to translate to and from Anthropic Messages. That hub keeps tier decisions, response-side premium escalation, and routed accounting shared across client formats.

Codex Responses is the intentional exception on the routed serving path. Codex traffic can route to provider-native Responses endpoints, especially ChatGPT OAuth-backed Codex endpoints, and those requests must preserve prefix-cache byte determinism. Routing the Codex request body through Anthropic conversion purely to reuse accounting would add a body round trip and could change serialized prefixes, lowering cache hit rate.

## Decision

Keep Anthropic Messages as the canonical internal contract for adapters and the generic routed core. Let Codex/Responses bypass Anthropic conversion only for its provider-native routed call, while normalizing response usage into the shared routed recording tail. Share accounting and premium-tool escalation decisions; do not share Codex request bytes through Anthropic conversion.

## Consequences

Codex request bytes sent upstream remain a deterministic function of the incoming Responses body with routed-only reasoning stripped. Routed Codex and routed Anthropic/DeepSeek calls write the same ledger and metrics shape through `record_route_outcome`. Future cleanup may reduce server size, but must not replace the Codex bypass with Responses-to-Anthropic-to-Responses conversion unless cache determinism is proven unchanged.
