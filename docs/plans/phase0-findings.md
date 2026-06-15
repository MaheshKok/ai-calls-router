# Phase 0 Findings: OpenAI Chat & Responses API contracts, Codex statefulness, wire mode, and tool vocabulary

Status: COMPLETE — every placeholder in `docs/plans/openai-compatibility.md` is resolved below.
Artifacts: this document only (Phase 0 is "no code").

---

## 1. OpenAI Chat Completions — tool-call contract (verified)

Source: official `openai-python` repo (SDK types generated from the OpenAPI spec by Stainless), cloned at
`/tmp/openai-python` (main branch, fresh `git clone --depth 1`).

### 1.1 Assistant message tool-calls (response)

File: `src/openai/types/chat/chat_completion_message.py`
File: `src/openai/types/chat/chat_completion_message_function_tool_call.py`

```python
# ChatCompletionMessage (response) — lines 57–89
class ChatCompletionMessage(BaseModel):
    content: Optional[str] = None
    refusal: Optional[str] = None
    role: Literal["assistant"]
    tool_calls: Optional[List[ChatCompletionMessageToolCallUnion]] = None
    function_call: Optional[FunctionCall] = None  # deprecated, replaced by tool_calls

# ChatCompletionMessageFunctionToolCall — lines 25–35
class ChatCompletionMessageFunctionToolCall(BaseModel):
    id: str            # "The ID of the tool call."
    function: Function # { arguments: str, name: str }
    type: Literal["function"]
```

**Key fact:** `tool_calls[].id` is always present when the model calls a tool. `function.arguments` is a **JSON string** (not a parsed object). The adapter must pass `id` verbatim through to Anthropic `tool_use.id` and back.

### 1.2 Tool result message (request)

File: `src/openai/types/chat/chat_completion_tool_message_param.py`

```python
class ChatCompletionToolMessageParam(TypedDict, total=False):
    content: Required[Union[str, Iterable[ChatCompletionContentPartTextParam]]]
    role: Required[Literal["tool"]]
    tool_call_id: Required[str]   # "Tool call that this message is responding to."
```

**Key fact:** Tool results use `role: "tool"` with `tool_call_id` matching the assistant tool-call's `id`. The adapter maps `role:"tool"` → Anthropic `tool_result` block with `tool_use_id = tool_call_id`.

### 1.3 Tool definitions in the request

File: `src/openai/types/chat/chat_completion_function_tool_param.py`

```python
class ChatCompletionFunctionToolParam(TypedDict, total=False):
    function: Required[FunctionDefinition]
    type: Required[Literal["function"]]
```

`FunctionDefinition` is at `src/openai/types/shared_params/function_definition.py` and carries `name`, `description`, `parameters` (JSON Schema). This maps directly to the inverse of the existing `convert_anthropic_tool` at `ai_calls_router/_lib/conversion.py:34`.

### 1.4 Streaming SSE chunks

File: `src/openai/types/chat/chat_completion_chunk.py`

```python
class ChatCompletionChunk(BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: Literal["chat.completion.chunk"]

class Choice(BaseModel):
    delta: ChoiceDelta
    finish_reason: Optional[Literal["stop","length","tool_calls","content_filter","function_call"]]
    index: int

class ChoiceDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[Literal["developer","system","user","assistant","tool"]] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None

class ChoiceDeltaToolCall(BaseModel):
    index: int           # index of the tool call in the array
    id: Optional[str] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None  # { arguments, name }
    type: Optional[Literal["function"]] = None
```

**Key fact:** The adapter must accumulate tool-call deltas across multiple chunks into complete `tool_calls[]` entries in the final `chat.completion` object. `[DONE]` terminates the SSE stream.

### 1.5 Full request body

File: `src/openai/types/chat/completion_create_params.py`

Key fields: `model`, `messages` (array of message params), `tools` (array of tool params), `tool_choice`, `stream` (boolean), `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stop`, `parallel_tool_calls`.

---

## 2. OpenAI Responses API — contract (verified)

Source: OpenAI Codex CLI source, cloned at `/tmp/codex-src` (HEAD `dfd03ea`).

### 2.1 Request body (`POST /v1/responses`)

File: `codex-rs/codex-api/src/common.rs:183–203`

```rust
pub struct ResponsesApiRequest {
    pub model: String,
    pub instructions: String,       // system prompt
    pub input: Vec<ResponseItem>,   // conversation items
    pub tools: Vec<serde_json::Value>,
    pub tool_choice: String,        // "auto"
    pub parallel_tool_calls: bool,
    pub reasoning: Option<Reasoning>,
    pub store: bool,                // server-side state (Azure only for Codex)
    pub stream: bool,               // always true for Codex
    pub include: Vec<String>,
    pub service_tier: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub text: Option<TextControls>, // verbosity control
    pub client_metadata: Option<HashMap<String, String>>,
}
```

### 2.2 `input[]` item types

File: `codex-rs/protocol/src/models.rs:666–700` (`ResponseInputItem` enum)
File: `codex-rs/protocol/src/models.rs:755–904` (`ResponseItem` enum)

**ResponseInputItem** types (items sent BY the client in the request):
- `Message { role, content }` — user/system message
- `FunctionCallOutput { call_id, output }` — tool result for function calls
- `McpToolCallOutput { call_id, output }` — MCP tool result
- `CustomToolCallOutput { call_id, name?, output }` — freeform tool result
- `ToolSearchOutput { call_id, status, execution, tools }` — tool search result

**ResponseItem** types (items returned BY the API and also used internally by Codex):
- `Message { id?, role, content, phase? }`
- `AgentMessage { author, recipient, content }`
- `Reasoning { id, summary, content?, encrypted_content? }`
- `LocalShellCall { id?, call_id?, status, action }`
- `FunctionCall { id?, name, namespace?, arguments, call_id }`
- `ToolSearchCall { id?, call_id?, status?, execution, arguments }`
- `FunctionCallOutput { call_id, output }`
- `CustomToolCall { id?, status?, call_id, name, input }`
- `CustomToolCallOutput { call_id, name?, output }`
- `ToolSearchOutput { call_id?, status, execution, tools }`
- `WebSearchCall { id?, status?, action? }`
- `ImageGenerationCall { id, status, revised_prompt?, result }`
- `Compaction`, `CompactionTrigger`, `ContextCompaction`

**Key fact:** `FunctionCallOutput` and `CustomToolCallOutput` use `call_id` (not `tool_call_id`) to link to the original function/tool call. The adapter maps `FunctionCallOutput` → Anthropic `tool_result` with `tool_use_id = call_id`.

### 2.3 Tool definitions in the Responses API

File: `codex-rs/tools/src/tool_spec.rs:17–51`

Tools are sent as JSON objects of these top-level `type` values:
- `"function"` — function-call tools (`ResponsesApiTool`: name, description, strict, parameters JSON Schema)
- `"namespace"` — namespaced tool groups (`ResponsesApiNamespace`)
- `"tool_search"` — tool search
- `"image_generation"` — image generation (`output_format`)
- `"web_search"` — web search with filters/location/context_size
- `"custom"` — freeform tools (`FreeformTool`: name, description, format grammar)

---

## 3. Codex statefulness (decision-critical — RESOLVED)

**Finding: Codex sends full `input[]` history. The router CAN see it.**

Evidence chain:

1. `build_responses_request` at `codex-rs/core/src/client.rs:720–773` calls:
   ```rust
   let input = prompt.get_formatted_input_for_request(model_info.use_responses_lite);
   ```
2. `get_formatted_input_for_request` at `codex-rs/core/src/client_common.rs:57–66` simply clones `self.input` (strips image details when `use_responses_lite` is true, but does not truncate history):
   ```rust
   fn get_formatted_input_for_request(&self, use_responses_lite: bool) -> Vec<ResponseItem> {
       let mut input = self.input.clone();
       if use_responses_lite {
           strip_image_details(&mut input);
       }
       input
   }
   ```
3. `store` field in `ResponsesApiRequest` at `codex-rs/core/src/client.rs:765` is set to:
   ```rust
   store: provider.is_azure_responses_endpoint(),
   ```
   This is `true` only for Azure, **false** for the standard OpenAI provider.

4. The `ResponsesApiRequest` struct at `codex-rs/codex-api/src/common.rs:183–203` does NOT include a `previous_response_id` field — the request is self-contained with the full `input[]` array.

**Conclusion:** Codex sends the full conversation history via `input[]` every turn and does not rely on `previous_response_id` server-side state. The router can inspect all `input[]` items to extract pending tool-output call_ids and match them to function-call names → **Codex routing is feasible.**

**Caveat:** The Azure Responses endpoint uses `store: true` and may maintain server-side state. Codex routing for Azure is not supported in this plan; document this as a limitation.

---

## 4. Codex wire mode (scope lever — RESOLVED)

**Finding: Codex uses the Responses API only. Chat Completions wire mode is removed.**

Evidence:

1. `WireApi` enum at `codex-rs/model-provider-info/src/lib.rs:50–57`:
   ```rust
   pub enum WireApi {
       #[default]
       Responses,
   }
   ```
2. Deserialization at lines 68–80: `"chat"` returns a hard error:
   ```rust
   "chat" => Err(serde::de::Error::custom(CHAT_WIRE_API_REMOVED_ERROR)),
   ```
3. `CHAT_WIRE_API_REMOVED_ERROR` at line 46:
   ```
   `wire_api = "chat"` is no longer supported.
   How to fix: set `wire_api = "responses"` in your provider config.
   More info: https://github.com/openai/codex/discussions/7782
   ```
4. `create_openai_provider` at line 324 sets `wire_api: WireApi::Responses`.
5. The user's local Codex config (`~/.codex/config.toml`) has no custom `model_providers` block — it uses the default OpenAI provider which speaks Responses.

**Decision: Phase 4 (Responses adapter) is IN, not OUT.** The router must implement `POST /v1/responses`.

---

## 5. Codex tool vocabulary and tier assignments

Source: Codex source code at `/tmp/codex-src`.

### 5.1 Complete Codex built-in tool vocabulary

From the tool dispatch map at `codex-rs/rollout-trace/src/tool_dispatch.rs:261–277`,
the tool specification files, and the test suite:

| Tool name | Type | Description | Source |
|-----------|------|-------------|--------|
| `exec_command` / `local_shell` / `shell` / `shell_command` | Function (`ResponsesApiTool`) | Execute shell commands | `tool_dispatch.rs:263` |
| `write_stdin` | Function | Write to a running process's stdin | `tool_dispatch.rs:264` |
| `apply_patch` | Freeform (grammar-based) | Edit files with patch format | `apply_patch_spec.rs`, `tool_dispatch.rs:265` |
| `view_image` | Function | View/analyze images | `view_image.rs:28-29` |
| `update_plan` | Function | Update the task plan | `plan_spec.rs`, `tool_dispatch.rs` (via freeform exec) |
| `web_search` / `web_search_preview` | Hosted `web_search` | Web search tool | `tool_spec.rs:36`, `tool_dispatch.rs:266` |
| `image_generation` / `image_query` | Hosted `image_generation` | Image generation | `tool_spec.rs:28`, `tool_dispatch.rs:267` |
| `spawn_agent` | Function (multi-agent) | Spawn a sub-agent | `tool_dispatch.rs:268` |
| `send_message` | Function (multi-agent) | Send message to agent | `tool_dispatch.rs:269` |
| `followup_task` / `assign_task` | Function (multi-agent) | Assign follow-up tasks | `tool_dispatch.rs:270` |
| `wait_agent` | Function (multi-agent) | Wait for agent completion | `tool_dispatch.rs:271` |
| `close_agent` / `interrupt_agent` | Function (multi-agent) | Close/interrupt agent | `tool_dispatch.rs:272` |
| `request_user_input` | Function | Ask the user for input | `model_runtime_selectors.rs:176` |
| `get_context_remaining` | Function | Query context window budget | `get_context_remaining_spec.rs:8` |
| `exec` (code-mode entrypoint) | Freeform (grammar-based) | Code-mode execution (bundles many tool calls) | `code-mode-protocol/src/lib.rs:44` |
| `wait` (code-mode) | Function | Wait for code-mode cell | `code-mode-protocol/src/lib.rs:45` |
| `new_context_window` | Function | Open a fresh context window | `new_context_window.rs:10-11` |
| `list_agents` | Function (multi-agent) | List spawned agents | `tool_dispatch.rs` (in `multi_agent_v1` namespace) |
| `request_plugin_install` | Function | Install Codex plugins | `request_plugin_install.rs:18-19` |
| `request_permissions` | Function | Request permissions | `request_permissions.rs:16-17` |

**Note on Codex tool naming in Responses wire format:**
- Function tools appear as `type: "function"` with `name` field in the tools array.
- Freeform tools appear as `type: "custom"` with `name` field.
- Hosted tools appear as `type: "web_search"` or `type: "image_generation"`.
- When Codex dispatches a `function_call`, the `call_id` links to the specific call; results use `FunctionCallOutput { call_id, output }`.

**Note on `tool_search` and deferred tools:** Codex supports a `tool_search` mechanism where tools can be loaded on demand. The `defer_loading` flag on `ResponsesApiTool` controls this. For routing purposes, all function-call names that appear in `function_call_output.call_id` lookups are relevant.

### 5.2 Tier assignments for Codex tools

Following the same tier taxonomy as the current Claude Code map (`fast`, `code`, `crud`, `premium`):

| Tier | Codex tools | Rationale |
|------|-------------|-----------|
| **fast** | `exec_command`, `local_shell`, `shell`, `shell_command`, `write_stdin`, `view_image` | Stateless command execution — the high-volume "interpret shell output" turn; cheap models handle it well |
| **code** | _(none native)_ | CORRECTED. Codex has NO built-in code-introspection tools. It reads/searches via the shell (`exec_command`) and edits via `apply_patch`. The only `read_file` in source is a TEST mock of an MCP filesystem handler (`core/src/tools/handlers/mcp.rs:451`). A `code` tier only fires if a filesystem MCP (e.g. `filesystem.read_file`) is wired into Codex — opt-in/empty by default. |
| **crud** | `update_plan`, `get_context_remaining`, `list_agents`, `wait_agent`, `request_permissions` | CRUD/state-management operations with low complexity |
| **premium** | `apply_patch`, `spawn_agent`, `send_message`, `followup_task`, `assign_task`, `close_agent`, `interrupt_agent`, `request_user_input`, `new_context_window`, `request_plugin_install`, `exec`, `wait` | Code modification, multi-agent orchestration, security-sensitive operations, user interaction |

**Premium rationale detail:**
- `apply_patch`: modifies code — same reasoning as Claude Code `Edit`, `Write`, `MultiEdit`
- `spawn_agent` / `send_message` / `followup_task` / `assign_task` / `close_agent` / `interrupt_agent`: multi-agent orchestration requires precise instruction-following
- `request_user_input`: user-facing, cannot be wrong
- `new_context_window`: structural operation that can silently lose context
- `request_plugin_install`: security boundary — must not be routed to an untrusted model
- `exec` / `wait`: code-mode entrypoints that aggregate other tool capabilities

**Note:** `web_search` and `image_generation` are hosted tools (type `"web_search"` and `"image_generation"` in the tools array) — Codex executes them differently. Their results appear as `WebSearchCall` and `ImageGenerationCall` response items, not as `function_call_output`. The adapter must handle these custom item types when extracting pending tools.

---

## 6. Hermes tool map, premium upstream, and provider

### 6.1 Hermes tool map (from the ACTUAL Hermes tool router) — CORRECTED

Prior draft sourced this from `config.example.yaml` (ai-calls-router's OWN
config), which is circular: it proves nothing about Hermes. Ground truth is the
live Hermes router that ai-calls-router replaces:
`/Users/maheshkokare/.hermes/plugins/tool-router/routes.yaml`.

Hermes' tool vocabulary is **distinct** from Claude Code's — do NOT reuse the
Claude map (`Bash`/`Read`/`Edit`/...). Hermes verbs:

| Hermes tool | Tier |
|-------------|------|
| `terminal` | fast |
| `process` | fast |
| `read_file` | code |
| `search_files` | code |
| `execute_code` | code |
| `skill_view` | code |
| `todo` | crud |
| `memory` | crud |
| `session_search` | crud |
| `skills_list` | crud |
| `write_file` | structured |
| `skill_manage` | structured |
| `cronjob` | structured |
| `patch` | premium |
| `clarify` | premium |
| `delegate_task` | premium |
| `browser_vision` | premium |
| `browser_*` | premium (trailing-`*` glob) |

- `premium_tools` (escalation guard): `[patch, clarify, delegate_task]`.
- `tier_precedence`: `[premium, structured, code, fast, crud]`.
- **New tier:** Hermes uses a **`structured`** tier (`write_file`,
  `skill_manage`, `cronjob`) that does NOT exist in ai-calls-router's current
  `tiers:` (only `fast`/`code`/`crud`). The `hermes` agent group therefore
  requires a `structured` tier added to `tiers:` (or remapped onto an existing
  tier) — see §8.

### 6.2 Hermes wire format, premium upstream, and provider — RESOLVED

The open "Chat vs Responses" question is settled by scanning the Hermes agent
codebase (`/Users/maheshkokare/.hermes/hermes-agent`). **Hermes has no single
inbound wire — `api_mode` is per-provider and Hermes natively emits all three.**

Ground truth — `hermes_cli/runtime_provider.py`:
- `_VALID_API_MODES` (`:240-251`): `chat_completions`, `codex_responses`,
  `anthropic_messages`, `bedrock_converse`, `codex_app_server`.
- Per-provider resolution (`_resolve_runtime_from_pool_entry`, `:308-341`):
  - `openai-codex`, `xai`/`xai-oauth` → **`codex_responses`**
  - `qwen-oauth`, `google-gemini-cli`, `nous`, `openrouter`, and the DEFAULT
    fallthrough (`:308`) → **`chat_completions`**
  - `anthropic`, `minimax-oauth` → **`anthropic_messages`**
  - `copilot`/`azure-foundry` → detected/configured
- Config schema confirms it (`hermes_cli/config.py:1751-1752`):
  `api_mode` is `"chat_completions" | "codex_responses" | "anthropic_messages"`,
  empty = auto-detect from base_url (`_detect_api_mode_for_url`), default
  `chat_completions`.

**Consequence — both Phase 3 and Phase 4 have real Hermes consumers; neither is
droppable:**
- A Hermes session on a chat-completions provider (qwen/gemini/nous/openrouter,
  or any custom OpenAI-compatible provider — the default) → hits ai-calls-router
  as **Chat Completions** → needs **Phase 3**.
- A Hermes session on `openai-codex`/`xai` → hits as **Responses** → needs
  **Phase 4** (same adapter as Codex).
- A Hermes session on Anthropic/minimax → **Anthropic Messages** → already served
  by the existing `/v1/messages` path (Phase 1).

Which one a given user sends depends on the provider/model they configure Hermes
to talk to ai-calls-router with. To be universal for Hermes, the router must
accept all three. (`bedrock_converse` is a 4th wire = AWS Bedrock; `codex_app_server`
hands the whole turn to a Codex subprocess, not an HTTP model wire — both
OUT-OF-SCOPE for v1, passthrough-only, same treatment as Azure Codex.)

- **Premium upstream:** the **Hermes session model** — "premium is implicit … it
  is whatever model the Hermes session runs on" (routes.yaml:5-6). It is NOT a
  fixed `https://api.anthropic.com` endpoint, and NOT necessarily Anthropic.
  The premium provider/wire equals the session's own provider/`api_mode`.
- routes.yaml is the tier-routing config (Hermes → downstream grunt provider). It
  pins the tool map / premium list / tiers; the inbound wire is set by the Hermes
  session's provider per the runtime_provider.py rules above.

When the plan creates the `hermes` agent group entry, seed `agent_defaults.py`
from the §6.1 map (with the `structured` tier), premium_tools
`[patch, clarify, delegate_task]`, and a Hermes-session premium upstream — not
the Claude Code map.

---

## 7. Decisions recorded

| Decision | Status | Record |
|----------|--------|--------|
| D1. Codex wire mode | RESOLVED | Responses API ONLY. Phase 4 is IN. |
| D2. Codex statefulness | RESOLVED | Full `input[]` history sent each turn. Codex routing is feasible. Azure caveat documented. |
| D3. Hermes wire details | RESOLVED | Tool map / premium list / tiers GROUNDED in the real `routes.yaml` (distinct vocab — `terminal`/`read_file`/`patch`/...; adds a `structured` tier; premium=`[patch, clarify, delegate_task]`). Wire format is **per-provider, multi-wire** (`hermes-agent` `runtime_provider.py:240-341`): Hermes emits `chat_completions` (default + qwen/gemini/nous/openrouter), `codex_responses` (openai-codex/xai), or `anthropic_messages` (anthropic/minimax). **Both Phase 3 (Chat) and Phase 4 (Responses) have real Hermes consumers — neither is droppable.** Premium upstream = Hermes session model (its own provider/wire). Bedrock/codex_app_server out of scope. |
| D4. Per-tool dup-line reduction gating | CARRIED OVER | Not resolved here; tracked separately. Default is OFF. |

---

## 8. Implications for later phases

1. **Phase 1 (adapter abstraction):** No change from plan. The `ClientAdapter` Protocol and `AnthropicMessagesAdapter` are wire-format agnostic and ready for Phases 3/4.

2. **Phase 2 (per-agent tool config):** `agent_defaults.py` now has real data for all three groups:
   - `claude_code`: existing map from `config.example.yaml:85–104`
   - `hermes`: real map from `routes.yaml` (§6.1) — distinct vocab, NOT the Claude map; premium_tools `[patch, clarify, delegate_task]`; premium upstream = Hermes session model (provider `openai-codex`). **Requires a `structured` tier** (`write_file`/`skill_manage`/`cronjob`) that the current `tiers:` lacks: either add `structured` to the config `tiers:` schema or remap those three onto `code`/`crud`. The per-agent config schema must allow tier names beyond `fast`/`code`/`crud`.
   - `codex`: the Codex tool vocabulary from §5 above. Routable surface is thin — `shell` family → `fast`, `apply_patch` → `premium`, multi-agent/`request_user_input`/`request_plugin_install` → `premium`, `update_plan`/`get_context_remaining` → `crud`. **No native `code` tier** (Codex reads/searches via the shell), so the `code` tier is empty for codex unless a filesystem MCP is wired.

3. **Phase 3 (Chat Completions adapter) — IN (real consumer confirmed):** A Hermes session on any chat-completions provider (qwen/gemini/nous/openrouter or any custom OpenAI-compatible provider — the DEFAULT `api_mode`) hits ai-calls-router as Chat Completions (D3, `runtime_provider.py:308-341`). Phase 3 is NOT droppable. The Chat schema is verified (§1); the adapter must:
   - Map `tool_calls[].id` ↔ Anthropic `tool_use.id` (verbatim)
   - Map `role:"tool"` with `tool_call_id` ↔ Anthropic `tool_result` with `tool_use_id`
   - Accumulate SSE tool-call deltas into complete `tool_calls[]`
   - Preserve byte stability for the DeepSeek prefix cache

4. **Phase 4 (Responses adapter — confirmed IN):** The Responses schema is verified (§2). The adapter must:
   - Map `FunctionCallOutput { call_id }` ↔ Anthropic `tool_result` with `tool_use_id = call_id`
   - Handle `CustomToolCallOutput`, `ToolSearchOutput`, `WebSearchCall`, `ImageGenerationCall` items
   - Handle the `input[]` array (not `messages[]`)
   - Extract pending tools from `FunctionCallOutput.call_id` matches to `FunctionCall.name` in `input[]`. **Hosted-tool items** (`WebSearchCall`, `ImageGenerationCall`) are NOT `function_call_output` and carry no routable function name — they are inert for tier selection; do not crash on them.
   - **`apply_patch` is a custom/freeform tool:** its result can arrive as `CustomToolCallOutput`. Name-matching for the premium-escalation guard must cover custom-tool call/output items, not only `FunctionCall`/`FunctionCallOutput`.
   - **Strip inbound Reasoning items** (Codex `Reasoning` items carry `encrypted_content`) before forming the DeepSeek-canonical prefix — mirror `_strip_thinking_from_messages` in `engine.py:35` — or byte-stability for the prefix cache breaks.
   - **Streaming is mandatory:** Codex ALWAYS streams; the live egress is the Responses SSE synthesizer (`response.created`, `response.output_item.added`, `response.function_call_arguments.delta`, …). A non-streaming JSON path is dead for codex.
   - Map `instructions` → Anthropic `system`

5. **Phase 5 (per-agent upstream):** Config carries per-agent upstream targets as planned.

6. **Azure Codex:** Not supported for routing. Document that Codex routing requires the standard OpenAI provider (not Azure responses endpoint).

---

## 9. Source citation index

| Source | Location | What it proves |
|--------|----------|----------------|
| OpenAI Python SDK types | `/tmp/openai-python/src/openai/types/chat/` (multiple files) | Chat Completions schema: assistant tool_calls with `id` + `function.{name,arguments}`, tool result messages with `role:"tool"` + `tool_call_id`, SSE chunk deltas with `ChoiceDeltaToolCall` |
| Codex client.rs | `/tmp/codex-src/codex-rs/core/src/client.rs:720–773` | Responses API request builder; `store` field set to `provider.is_azure_responses_endpoint()` |
| Codex client_common.rs | `/tmp/codex-src/codex-rs/core/src/client_common.rs:57–66` | Full `input[]` is sent every turn (cloned from `Prompt.input`); no truncation to `previous_response_id` |
| Codex codex-api common.rs | `/tmp/codex-src/codex-rs/codex-api/src/common.rs:183–203` | Full `ResponsesApiRequest` struct shape; no `previous_response_id` field |
| Codex model-provider-info lib.rs | `/tmp/codex-src/codex-rs/model-provider-info/src/lib.rs:46–80` | `WireApi` enum is Responses-only; `"chat"` returns hard error; Chat mode is removed |
| Codex models.rs | `/tmp/codex-src/codex-rs/protocol/src/models.rs:666–700, 755–904` | `ResponseInputItem` and `ResponseItem` enum variants — all input/output item types |
| Codex tool_spec.rs | `/tmp/codex-src/codex-rs/tools/src/tool_spec.rs:17–51` | `ToolSpec` enum — function, namespace, tool_search, image_generation, web_search, custom (freeform) tool types |
| Codex tool_dispatch.rs | `/tmp/codex-src/codex-rs/rollout-trace/src/tool_dispatch.rs:261–277` | Full Codex built-in tool name map (`exec_command`, `local_shell`, `shell`, `shell_command`, `write_stdin`, `apply_patch`, `web_search`, `web_search_preview`, `image_generation`, `image_query`, `spawn_agent`, `send_message`, `followup_task`, `assign_task`, `wait_agent`, `close_agent`, `interrupt_agent`) |
| Codex plan_spec.rs | `/tmp/codex-src/codex-rs/core/src/tools/handlers/plan_spec.rs` | `update_plan` tool specification |
| Codex apply_patch_spec.rs | `/tmp/codex-src/codex-rs/core/src/tools/handlers/apply_patch_spec.rs` | `apply_patch` freeform tool specification |
| Codex code-mode-protocol lib.rs | `/tmp/codex-src/codex-rs/code-mode-protocol/src/lib.rs:44–45` | `exec` (PUBLIC_TOOL_NAME) and `wait` (WAIT_TOOL_NAME) code-mode tool names |
| Codex local config | `/Users/maheshkokare/.codex/config.toml` | No custom `model_providers` block — confirms default OpenAI provider with Responses wire API |
| Hermes tool router (GROUND TRUTH) | `/Users/maheshkokare/.hermes/plugins/tool-router/routes.yaml:33-95` | Real Hermes tiers (`fast`/`code`/`crud`/`structured`), tool→tier map (`terminal`/`read_file`/`patch`/...), `premium_tools: [patch, clarify, delegate_task]`, `tier_precedence`, provider `openai-codex` + `api_mode: codex_responses` (Hermes is OpenAI-style, premium = session model) |
| Hermes agent runtime (GROUND TRUTH) | `/Users/maheshkokare/.hermes/hermes-agent/hermes_cli/runtime_provider.py:240-341`, `hermes_cli/config.py:1751-1752` | Hermes inbound wire is per-provider multi-wire: `_VALID_API_MODES` = {chat_completions, codex_responses, anthropic_messages, bedrock_converse, codex_app_server}; default `chat_completions`; openai-codex/xai→codex_responses; anthropic/minimax→anthropic_messages. Proves Phase 3 AND Phase 4 both have Hermes consumers (D3 resolved). |
| Codex MCP filesystem test mock | `/tmp/codex-src/codex-rs/core/src/tools/handlers/mcp.rs:451` | The ONLY `read_file` in Codex source is a test mock — proves Codex has no native code-introspection tools (Defect 1 evidence) |
| ai-calls-router config.example.yaml | `config.example.yaml:85–104` | ai-calls-router's OWN Claude-Code tool map (NOT a Hermes source — prior draft's circular citation) |
