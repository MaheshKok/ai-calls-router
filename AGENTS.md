<!-- lean-ctx -->
  ## lean-ctx

  Prefer lean-ctx for token-compressed repo inspection.

  Use:

  ```bash
  lean-ctx -c "git status --short"
  lean-ctx -c "git diff --stat"
  lean-ctx -c "git diff --cached"
  lean-ctx -c "sed -n '1,200p' AGENTS.md"
  lean-ctx -c "rg -n 'pattern' path"

  Rules:

  - Use lean-ctx -c "<command>" for shell commands that may
    produce large output.

  - Do not wrap commands with rtk.
  - If lean-ctx blocks a command because of its allowlist,
    either allow the exact command with lean-ctx allow
    <cmd> or run the raw command only when debugging.

  - Keep raw commands for small, exact output where
    compression would hide formatting.
  <!-- /lean-ctx -->
