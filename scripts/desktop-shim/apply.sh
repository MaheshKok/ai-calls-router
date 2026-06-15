#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_SRC="$SCRIPT_DIR/claude-shim.zsh"
MARKER="ai-calls-router desktop shim"
BASE="$HOME/Library/Application Support/Claude/claude-code"
ACR_URL="${ACR_DESKTOP_SHIM_URL:-http://127.0.0.1:8747}"
STAMP="$(date '+%Y-%m-%d %H:%M:%S')"

usage() {
  cat <<'EOF'
Usage:
  apply.sh              install or refresh the Desktop shim
  apply.sh --revert     restore original Claude Code binaries
  apply.sh --status     report install state and current target
EOF
}

status_mode=0
revert=0
if [[ "${1:-}" == "--status" ]]; then
  status_mode=1
elif [[ "${1:-}" == "--revert" ]]; then
  revert=1
elif [[ -n "${1:-}" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$BASE" ]]; then
  echo "[$STAMP] acr-desktop-shim: no Claude Desktop claude-code directory found: $BASE" >&2
  exit 0
fi

report_pointing() {
  local macos_dir="$1"
  local bin="$macos_dir/claude"
  local real="$macos_dir/claude-real"

  if [[ ! -e "$bin" ]]; then
    echo "points to: missing claude binary"
    return
  fi

  if [[ -f "$bin" ]] && grep -q "$MARKER" "$bin" 2>/dev/null; then
    if [[ -f "$real" ]]; then
      echo "points to: $ACR_URL"
    else
      echo "points to: shim present but claude-real missing"
    fi
  elif [[ -f "$bin" ]] && /usr/bin/file "$bin" | grep -q "Mach-O"; then
    echo "points to: original Claude binary"
  elif [[ -f "$real" ]]; then
    echo "points to: other shim or damaged install"
  else
    echo "points to: unknown"
  fi
}

if (( status_mode )); then
  echo "acr-desktop-shim status"
  echo "base: $BASE"
  echo "target proxy: $ACR_URL"
  if /usr/bin/curl -s -m 1 -o /dev/null "$ACR_URL/health" 2>/dev/null; then
    echo "proxy health: ok"
  else
    echo "proxy health: down"
  fi

  found=0
  for macos_dir in "$BASE"/*/claude.app/Contents/MacOS; do
    [[ -d "$macos_dir" ]] || continue
    found=1
    echo "version dir: ${macos_dir%/claude.app/Contents/MacOS}"
    bin="$macos_dir/claude"
    real="$macos_dir/claude-real"
    if [[ -f "$bin" ]] && grep -q "$MARKER" "$bin" 2>/dev/null && [[ -f "$real" ]]; then
      echo "installed: $macos_dir"
    elif [[ -f "$bin" ]] && /usr/bin/file "$bin" | grep -q "Mach-O"; then
      echo "unshimmed: $macos_dir"
    elif [[ -f "$real" ]]; then
      echo "other-shim-or-damaged: $macos_dir"
    else
      echo "missing-binary: $macos_dir"
    fi
    report_pointing "$macos_dir"
  done
  if (( ! found )); then
    echo "no desktop-managed Claude Code versions found"
  fi
  exit 0
fi

found=0
for macos_dir in "$BASE"/*/claude.app/Contents/MacOS; do
  [[ -d "$macos_dir" ]] || continue
  found=1
  bin="$macos_dir/claude"
  real="$macos_dir/claude-real"

  if (( revert )); then
    if [[ -f "$bin" ]] && grep -q "$MARKER" "$bin" 2>/dev/null && [[ -f "$real" ]]; then
      mv "$real" "$bin"
      echo "[$STAMP] acr-desktop-shim: reverted $macos_dir"
    elif [[ -f "$bin" ]] && /usr/bin/file "$bin" | grep -q "Mach-O"; then
      echo "[$STAMP] acr-desktop-shim: skipped $macos_dir (current claude is not the ACR shim; claude-real exists)" >&2
    fi
    continue
  fi

  if [[ -f "$bin" ]] && grep -q "$MARKER" "$bin" 2>/dev/null; then
    if [[ ! -f "$real" ]]; then
      echo "[$STAMP] acr-desktop-shim: ERROR shim present but claude-real missing in $macos_dir" >&2
      exit 1
    fi
    echo "[$STAMP] acr-desktop-shim: already installed $macos_dir"
    continue
  fi

  if [[ -f "$bin" ]] && /usr/bin/file "$bin" | grep -q "Mach-O"; then
    mv "$bin" "$real"
    cp "$SHIM_SRC" "$bin"
    chmod +x "$bin" "$real"
    echo "[$STAMP] acr-desktop-shim: installed $macos_dir"
  fi
done

if (( ! found )); then
  echo "[$STAMP] acr-desktop-shim: no Claude Desktop claude-code installation found under $BASE" >&2
  exit 1
fi
