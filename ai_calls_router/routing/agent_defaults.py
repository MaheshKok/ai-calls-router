"""Default agent tool maps for routing and wizard generation.

This module is data-only so the same defaults can be reused by config
validation, routing decisions, and the interactive wizard without importing
IO or behavior. Keep the values synchronized with the plan documents and the
wizard's generated config schema.
"""

from __future__ import annotations

AGENT_DEFAULT_TOOLS: dict[str, dict[str, str]] = {
    "claude_code": {
        "Bash": "fast",
        "BashOutput": "fast",
        "KillShell": "fast",
        "WebFetch": "fast",
        "WebSearch": "fast",
        "Read": "code",
        "Grep": "code",
        "Glob": "code",
        "LSP": "code",
        "TodoWrite": "crud",
        "TaskList": "crud",
        "TaskGet": "crud",
        "Edit": "premium",
        "Write": "premium",
        "MultiEdit": "premium",
        "NotebookEdit": "premium",
        "Task": "premium",
        "ExitPlanMode": "premium",
        "AskUserQuestion": "premium",
    },
    "hermes": {
        "terminal": "fast",
        "process": "fast",
        "read_file": "code",
        "search_files": "code",
        "execute_code": "code",
        "skill_view": "code",
        "todo": "crud",
        "memory": "crud",
        "session_search": "crud",
        "skills_list": "crud",
        "write_file": "structured",
        "skill_manage": "structured",
        "cronjob": "structured",
        "patch": "premium",
        "clarify": "premium",
        "delegate_task": "premium",
        "browser_vision": "premium",
        "browser_*": "premium",
    },
}

AGENT_DEFAULT_PREMIUM_TOOLS: dict[str, list[str]] = {
    "claude_code": [
        "Edit",
        "Write",
        "MultiEdit",
        "NotebookEdit",
        "Task",
        "ExitPlanMode",
        "AskUserQuestion",
    ],
    "hermes": ["patch", "clarify", "delegate_task"],
}
