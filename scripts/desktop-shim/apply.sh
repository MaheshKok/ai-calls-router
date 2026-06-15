#!/bin/zsh
# Installs, refreshes, or reverts the ai-calls-router shim for Claude Desktop's
# embedded Claude Code binary.
#
# Usage:
#   scripts/desktop-shim/apply.sh              install/refresh shim everywhere
#   scripts/desktop-shim/apply.sh --revert     restore original binaries
#   scripts/desktop-shim/apply.sh --force      replace another existing shim
#
# The shim patches every version directory under:
#   ~/Library/Application Support/Claude/claude-code/*/claude.app/Contents/MacOS
#
# It preserves the original Mach-O binary as `claude-real` and replaces `claude`
# with `claude-shim.zsh`. Re-run after Claude Desktop updates, because updates
# create a new version directory with a fresh unshimmed binary.

emulate -L zsh
set -u
setopt NULL_GLOB

SCRIPT_DIR="${0:A:h}"
SHIM_SRC="$SCRIPT_DIR/claude-shim.zsh"
MARKER="ai-calls-router desktop shim"
BASE="$HOME/Library/Application Support/Claude/claude-code"
ACR_URL="${ACR_DESKTOP_SHIM_URL:-http://127.0.0.1:8747}"
STAMP="$(date '+%Y-%m-%d %H:%M:%S')"

revert=0
force=0
status_mode=0

usage() {
  cat <<'EOF'
Usage:
  apply.sh              install/refresh the ACR Desktop shim
  apply.sh --status     show installed shim state and ACR health
  apply.sh --revert     restore original Claude Code binaries
  apply.sh --force      replace another existing shim when claude-real exists
  apply.sh --help       show this help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --revert)
      revert=1
      ;;
    --force)
      force=1
      ;;
    --status)
      status_mode=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[$STAMP] acr-desktop-shim: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -f "$SHIM_SRC" ]]; then
  echo "[$STAMP] acr-desktop-shim: ERROR missing shim source: $SHIM_SRC" >&2
  exit 1
fi

if [[ ! -d "$BASE" ]]; then
  echo "[$STAMP] acr-desktop-shim: no Claude Desktop claude-code directory found: $BASE" >&2
  exit 0
fi

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
    elif [[ -f "$real" ]]; then
      echo "[$STAMP] acr-desktop-shim: skipped $macos_dir (current claude is not the ACR shim; claude-real exists)" >&2
    else
      echo "[$STAMP] acr-desktop-shim: already unshimmed $macos_dir"
    fi
    continue
  fi

  if [[ -f "$bin" ]] && grep -q "$MARKER" "$bin" 2>/dev/null; then
    if [[ ! -f "$real" ]]; then
      echo "[$STAMP] acr-desktop-shim: ERROR shim present but claude-real missing in $macos_dir" >&2
      continue
    fi
    cp "$SHIM_SRC" "$bin" && chmod 755 "$bin"
    echo "[$STAMP] acr-desktop-shim: refreshed $macos_dir"
    continue
  fi

  if [[ -f "$bin" ]] && /usr/bin/file "$bin" | grep -q "Mach-O"; then
    mv "$bin" "$real" && cp "$SHIM_SRC" "$bin" && chmod 755 "$bin"
    echo "[$STAMP] acr-desktop-shim: installed in $macos_dir"
    continue
  fi

  if [[ -f "$bin" ]] && [[ -f "$real" ]] && (( force )); then
    cp "$SHIM_SRC" "$bin" && chmod 755 "$bin"
    echo "[$STAMP] acr-desktop-shim: force-replaced existing shim in $macos_dir"
    continue
  fi

  if [[ -f "$bin" ]] && [[ -f "$real" ]]; then
    echo "[$STAMP] acr-desktop-shim: skipped $macos_dir (another shim appears installed; use --force only if claude-real is the original binary)" >&2
    continue
  fi

  echo "[$STAMP] acr-desktop-shim: skipped $macos_dir (claude binary missing or not a Mach-O binary)" >&2
done

if (( ! found )); then
  echo "[$STAMP] acr-desktop-shim: no desktop-managed Claude Code versions found under $BASE" >&2
fi
