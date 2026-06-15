#!/usr/bin/env bash
set -euo pipefail

BASE="$HOME/Library/Application Support/Claude/claude-code"
TARGET_PORT="8747"
TARGET_PROXY="http://127.0.0.1:${TARGET_PORT}"

if [[ ! -d "$BASE" ]]; then
  echo "base: $BASE"
  echo "status: missing"
  exit 1
fi

found=0
shopt -s nullglob
for macos_dir in "$BASE"/*/claude.app/Contents/MacOS; do
  [[ -d "$macos_dir" ]] || continue
  found=1
  bin="$macos_dir/claude"
  real="$macos_dir/claude-real"

  echo "version dir: ${macos_dir%/claude.app/Contents/MacOS}"
  if [[ ! -e "$bin" ]]; then
    echo "state: missing claude binary"
    echo
    continue
  fi

  if [[ -f "$bin" ]] && grep -q 'ACR shim' "$bin" 2>/dev/null; then
    echo "state: shimmed"
    if [[ -f "$real" ]]; then
      if tr '\0' '\n' < "$bin" | grep -Fq "$TARGET_PROXY"; then
        echo "points to: $TARGET_PROXY"
      else
        echo "points to: shim present but proxy string not found"
      fi
    else
      echo "points to: shim present but claude-real missing"
    fi
  else
    echo "state: unshimmed"
    if [[ -x "$bin" ]]; then
      if file "$bin" 2>/dev/null | grep -q 'Mach-O'; then
        echo "points to: original Claude binary"
      else
        echo "points to: unknown file type"
      fi
    else
      echo "points to: not executable"
    fi
  fi
  echo
done

if [[ "$found" -eq 0 ]]; then
  echo "base: $BASE"
  echo "status: no Claude Code versions found"
  exit 1
fi
