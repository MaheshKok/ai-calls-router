#!/bin/zsh
# ai-calls-router desktop shim
#
# Routes Claude Desktop's embedded Claude Code sessions through the local ACR
# proxy. Claude Desktop injects ANTHROPIC_BASE_URL=https://api.anthropic.com
# into the Claude Code binary it launches, so shell/launchd environment changes
# are not enough. This shim is installed over:
#   ~/Library/Application Support/Claude/claude-code/*/claude.app/Contents/MacOS/claude
# and the real Mach-O binary is kept next to it as claude-real.
#
# Safety gates:
#   - Only the default Anthropic endpoint is rewritten. Custom endpoints pass
#     through untouched.
#   - ACR must answer /health within 1s. If ACR is down, the original endpoint
#     is kept so Desktop sessions still work.

ACR_URL="${ACR_DESKTOP_SHIM_URL:-http://127.0.0.1:8747}"
REAL_CLAUDE="${0:A:h}/claude-real"

if [[ ! -x "$REAL_CLAUDE" ]]; then
  cat >&2 <<EOF
acr-desktop-shim: missing executable next to shim: $REAL_CLAUDE

This file is a template copied into Claude Desktop's managed Claude Code app by:
  scripts/desktop-shim/apply.sh

Do not run scripts/desktop-shim/claude-shim.zsh directly from the repo. To check
whether the Desktop shim is installed, run:
  scripts/desktop-shim/apply.sh --status
EOF
  exit 127
fi

if [[ -z "${ANTHROPIC_BASE_URL:-}" || "${ANTHROPIC_BASE_URL}" == "https://api.anthropic.com" ]]; then
  if /usr/bin/curl -s -m 1 -o /dev/null "${ACR_URL}/health" 2>/dev/null; then
    export ANTHROPIC_BASE_URL="${ACR_URL}"
  fi
fi

exec "$REAL_CLAUDE" "$@"
