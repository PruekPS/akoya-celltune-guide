#!/bin/bash
# Check everything a run needs, before it costs anything.
#
#   source pipeline/preflight.sh <samples.csv> [conda-env]
#
# Three jobs died in their first ten seconds for want of this: a samples sheet
# whose name I guessed (41839280), a conda environment that exists on the Mac
# but not on HiPerGator (41840720), and a token file with a newline in it. Each
# failure was free to detect and expensive to discover -- one of them after a
# 2-13 minute image read. This is stage 00's gate, factored out so every job
# script can run it as its first act.
#
# Sets AKOYA_ENV and exits non-zero with a specific fix on any failure.
preflight() {
  local samples="${1:?usage: preflight <samples.csv> [env]}"
  local want_env="${2:-${AKOYA_ENV:-akoya-cellsam}}"
  local fail=0
  say() { printf '  %-6s %s\n' "$1" "$2"; }

  echo "pre-flight:"

  # 1. the samples sheet, named exactly
  if [ -f "$samples" ]; then
    say OK "samples sheet $samples ($(( $(wc -l < "$samples") - 1 )) rows)"
  else
    say FAIL "no samples sheet at '$samples'"
    echo "         available: $(ls samples/*.csv 2>/dev/null | tr '\n' ' ')"
    fail=1
  fi

  # 2. every image and panel the sheet points at
  if [ -f "$samples" ]; then
    python3 - "$samples" <<'PY' || fail=1
import csv, sys, os
bad = 0
with open(sys.argv[1], newline="") as fh:
    for row in csv.DictReader(fh):
        for col in ("image", "panel", "he_image", "channel_names"):
            p = (row.get(col) or "").strip()
            if not p:
                continue
            if os.path.exists(p):
                print(f"  OK     {col}: {os.path.basename(p)} ({os.path.getsize(p)/1e9:.2f} GB)")
            else:
                print(f"  FAIL   {col} missing for {row.get('sample_id','?')}: {p}")
                bad = 1
sys.exit(bad)
PY
  fi

  # 3. the conda environment, by name, before activating it
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$want_env"; then
    say OK "conda environment $want_env"
    export AKOYA_ENV="$want_env"
  else
    say FAIL "no conda environment named '$want_env' on this machine"
    echo "         available: $(conda env list 2>/dev/null | awk '!/^#/ && NF {print $1}' | tr '\n' ' ')"
    fail=1
  fi

  # 4. writable output space
  if [ -w . ]; then say OK "working directory is writable"; else say FAIL "cannot write to $(pwd)"; fail=1; fi

  if [ "$fail" -ne 0 ]; then
    echo "pre-flight FAILED - nothing was run." >&2
    return 1
  fi
  echo "pre-flight passed"
  return 0
}
