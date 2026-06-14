# Tool Result Routing Design: Current vs Proposed

This note documents the difference between the current `ai-calls-router` behavior and a proposed safer behavior where a cheap model processes tool output but Claude remains the primary planner and final decision-maker.

The goal is to make the routing trade-off easy to revisit when changing the router.

## Key terms

- **Claude / premium model**: The session model used by Claude Code / Claude Desktop. In this project, premium routing means passthrough to Anthropic.
- **Cheap model / routed model**: A configured lower-cost model such as `deepseek/deepseek-v4-pro` or `deepseek/deepseek-v4-flash`.
- **Tool use**: The assistant response that asks Claude Code to run a tool, for example `Grep(pattern="pricing|cost", path=".")`.
- **Tool result**: The next user message that contains the local output from the executed tool.
- **Tool-result routing**: Routing the follow-up model request based on which tool produced the result.

## Current routing baseline

The current implementation follows the same broad design as Hermes tool-router and Headroom tool-router:

1. Requests that do not contain pending tool results stay with the premium/session model.
2. Claude decides which tool to call.
3. Claude Code runs the tool locally.
4. The next request contains a `tool_result` block.
5. The router maps that `tool_result` back to the earlier `tool_use` name.
6. The tool name maps to a configured tier.
7. The tier maps to a model.
8. The routed model processes the follow-up request.
9. If the routed model attempts a premium/risky tool call, the router escalates/replays on Claude.

The important point: the current router routes the model call that processes the tool result. It does not route Claude's initial decision about which tool to call.

## Example task

User asks Claude Code:

> Find where request pricing is calculated and explain whether we should edit it.

Claude has tools available:

- `Grep`
- `Read`
- `Edit`
- and other Claude Code tools

Relevant routing config:

```yaml
tools:
  Grep: code
  Read: code
  Edit: premium

tiers:
  code:
    model: deepseek/deepseek-v4-pro
```

## Old/current behavior: tool-result routing

### End-to-end flow

1. User sends the request:

   ```text
   Find where request pricing is calculated and explain whether we should edit it.
   ```

2. The first model call goes to Claude/Anthropic.

   There is no tool result yet, so the router does not route to DeepSeek. Claude owns the initial plan and decides which tool to call.

3. Claude decides:

   ```text
   I should call Grep.
   ```

   Claude returns a tool call:

   ```text
   Grep(pattern="pricing|cost", path=".")
   ```

4. Claude Code runs `Grep` locally.

   Example tool output:

   ```text
   ai_calls_router/accounting/savings.py: register_tier_prices
   ai_calls_router/accounting/savings.py: record_routing_savings
   ```

5. Claude Code sends the next request to the proxy.

   That request contains:

   - the previous assistant `tool_use` named `Grep`
   - a new user `tool_result` containing the Grep output

6. The router resolves the tool name:

   ```text
   tool_result -> tool_use_id -> Grep
   ```

7. The config maps the tool to a tier:

   ```yaml
   Grep: code
   ```

8. The tier maps to a model:

   ```text
   code -> deepseek/deepseek-v4-pro
   ```

9. The router sends the whole follow-up request to DeepSeek.

10. DeepSeek now interprets the Grep result and decides the next assistant response.

    It may respond with natural language:

    ```text
    The pricing code is in ai_calls_router/accounting/savings.py. I should read it next.
    ```

    Or it may emit another tool call:

    ```text
    Read(file_path="ai_calls_router/accounting/savings.py")
    ```

11. If DeepSeek emits a premium/risky tool such as `Edit`, `Write`, `MultiEdit`, or `Task`, the router detects that and replays/escalates the request to Claude.

### What this means

After the `Grep` result comes back, DeepSeek becomes the model that interprets that result and chooses the next assistant action for that step, unless escalation occurs.

Claude made the original tool decision, but Claude may not directly inspect the raw `Grep` output before the next step.

### Pros of old/current behavior

- Cheaper: only one LLM call after tool output, usually to DeepSeek.
- Faster: no extra Claude replay after every cheap tool result.
- Matches existing Hermes tool-router and Headroom tool-router behavior.
- Good for simple tool-output-heavy loops:
  - `Grep`
  - `Read`
  - `BashOutput`
  - `LSP`
  - `TaskList`
  - `TaskGet`
- Keeps premium Claude for:
  - turn openers
  - unknown tools
  - configured premium tools
  - risky tool decisions if the cheap model escalates
- Simpler implementation.
- Lower chance of protocol bugs because the routed model response is simply returned as the assistant response.

### Cons of old/current behavior

- After a cheap tool result, DeepSeek becomes the next planner for that step.
- Claude does not necessarily inspect the raw tool output unless escalation happens.
- DeepSeek might make a lower-quality next decision than Claude.
- If DeepSeek gives final text, the final answer may come from DeepSeek, not Claude.
- Safety depends on the premium-tool escalation guard.
- For complex coding workflows, the cheap model may miss subtle repository intent or make weaker multi-step plans.

## New/proposed behavior: summarize/process with DeepSeek, then send back to Claude

The proposed behavior changes the role of the cheap model.

Instead of letting DeepSeek become the assistant after a cheap tool result, the router uses DeepSeek internally as a tool-output processor. The processed result is then sent back to Claude, and Claude remains the main planner and final decision-maker.

### End-to-end flow

1. User sends the request:

   ```text
   Find where request pricing is calculated and explain whether we should edit it.
   ```

2. The first model call still goes to Claude.

   Claude owns the initial planning step.

3. Claude decides to call `Grep`:

   ```text
   Grep(pattern="pricing|cost", path=".")
   ```

4. Claude Code runs `Grep` locally.

5. Claude Code sends the `tool_result` request to the proxy.

6. The router resolves the tool and tier:

   ```text
   Grep -> code -> deepseek/deepseek-v4-pro
   ```

7. Instead of letting DeepSeek become the assistant, the router calls DeepSeek in an internal helper mode with an instruction like:

   ```text
   Summarize and interpret this Grep output for Claude.
   Preserve important file paths, line numbers, uncertainty, and next-step hints.
   Do not call tools.
   ```

8. DeepSeek returns a compact interpretation:

   ```text
   Pricing appears in ai_calls_router/accounting/savings.py.
   Key functions: register_tier_prices and record_routing_savings.
   No edit yet; the next useful step is to read that file.
   ```

9. The router sends a new or modified request to Claude.

   Claude receives the original conversation plus either:

   - a summarized tool result, or
   - a synthetic helper output saying DeepSeek summarized the tool result

10. Claude decides the next action:

    ```text
    Read(file_path="ai_calls_router/accounting/savings.py")
    ```

11. Claude remains the main planner and final decision-maker.

### What this means

DeepSeek does not decide the next assistant action. It only helps digest noisy tool output.

Claude receives the processed evidence and continues planning from there.

### Pros of new/proposed behavior

- Claude remains the orchestrator after every tool result.
- Better safety for complex coding tasks.
- Better consistency with Claude Code's expected behavior.
- DeepSeek becomes a cheap tool-output processor instead of a planner.
- Good for huge noisy outputs:
  - `BashOutput`
  - long `Grep`
  - verbose `LSP`
  - large `TaskGet`
- Claude can make final decisions using compressed/summarized context.
- Risky decisions like `Edit`, `Write`, `MultiEdit`, and `Task` stay more naturally with Claude.
- Easier to reason about responsibility:
  - Claude chooses tools and final actions.
  - DeepSeek helps digest cheap tool output.

### Cons of new/proposed behavior

- More expensive than current routing:
  - one DeepSeek call plus one Claude call after each routed tool result
- Slower:
  - two model round trips after each routed tool result
- More implementation complexity:
  - need a new routing mode
  - need a summary prompt
  - need a schema for synthetic output
  - need fallback behavior
  - need tests around Anthropic message formatting
- Summaries can lose important details.
- If DeepSeek drops a key line from `Grep` or `BashOutput`, Claude may make a wrong decision from incomplete evidence.
- Harder to preserve exact tool semantics:
  - Anthropic tool protocol expects `tool_result` blocks tied to `tool_use_id`
  - replacing or wrapping them must be done carefully
- May hurt prompt-cache behavior if the proxy rewrites tool results differently every time.
- If always enabled, it may overuse Claude and reduce the cost savings this router is meant to achieve.

## Side-by-side comparison

| Dimension | Old/current: tool-result routing | New/proposed: summarize then Claude |
| --- | --- | --- |
| Who chooses the first tool? | Claude | Claude |
| Who processes cheap tool output? | DeepSeek as assistant | DeepSeek as internal helper |
| Who chooses the next action after `Grep`? | DeepSeek, unless escalation triggers | Claude |
| Number of model calls after routed tool result | Usually 1 | Usually 2 |
| Latency | Lower | Higher |
| Cost | Lower | Higher |
| Safety for complex code changes | Good only with escalation guard | Stronger because Claude remains planner |
| Protocol complexity | Lower | Higher |
| Risk of losing details | Lower for raw routed request; depends on DeepSeek reasoning | Higher if summarization drops details |
| Matches Hermes/Headroom behavior | Yes | No, this is a new mode |

## Suggested implementation direction

Keep the current behavior as the default because it matches Hermes and Headroom and provides the strongest cost/latency savings:

```yaml
settings:
  cheap_tool_result_mode: direct_response
```

Add the proposed behavior as an opt-in mode:

```yaml
settings:
  cheap_tool_result_mode: summarize_then_premium
```

A later refinement could make this per tool:

```yaml
tools:
  Grep:
    tier: code
    mode: summarize_then_premium
  BashOutput:
    tier: fast
    mode: summarize_then_premium
  Read:
    tier: code
    mode: direct_response
  Edit:
    tier: premium
```

This gives both behaviors:

- cheap direct routing for safe/simple tool-result turns
- Claude-owned decision-making for noisy, complex, or risky tool-result turns

## Open design questions before implementation

1. Should summarized tool output replace the original `tool_result`, or should it be added as a separate synthetic result?
2. Should Claude see the raw tool output, the summary, or both?
3. How should the router preserve `tool_use_id` links so Anthropic protocol semantics remain valid?
4. Should summarize-then-Claude be global, per tier, or per tool?
5. What is the fallback if DeepSeek fails while summarizing?
6. Should cache-sensitive requests avoid rewriting tool results to preserve prompt-cache hit rates?
7. Should the summary include a mandatory structured section with file paths, line numbers, uncertainty, and suggested next action?
