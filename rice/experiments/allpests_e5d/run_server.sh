#!/usr/bin/env bash
# Run every pest assigned to one server, sequentially (one GPU, one training at a time).
# Both servers run THIS script; only the number differs. There is no cross-server
# communication and no shared writable state -- the split is static in pests.tsv and every
# selection split is a pure hash of the sample id, so the two halves cannot diverge.
#
#   server 1:  bash rice/experiments/allpests_e5d/run_server.sh 1
#   server 2:  bash rice/experiments/allpests_e5d/run_server.sh 2
#
# Safe to re-run after an interruption: finished steps are skipped via their .done markers.
# A pest that fails does NOT abort the rest -- it is recorded and the sweep continues.
set -uo pipefail
AP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$AP/common.sh"

SERVER="${1:?usage: run_server.sh <1|2>}"
PESTS="$(pests_server "$SERVER")"
[ -z "$PESTS" ] && { echo "[server$SERVER] no pests assigned"; exit 2; }

mkdir -p "$OUT_ROOT/_run"
SUMMARY="$OUT_ROOT/_run/server${SERVER}_summary.tsv"
echo -e "pest\tstatus\tstarted\tfinished" > "$SUMMARY"

echo "===== server$SERVER: $(echo $PESTS | tr '\n' ' ') ====="
echo "[server$SERVER] pre-flight dry run (fast)"
cd "$CS"
if ! PYTHONPATH="$WS:$CS" $PY "$AP/dry_run.py" --level fast; then
  echo "[server$SERVER] ABORT: dry run reported FAIL -- fix before training"; exit 3
fi

rc_all=0
for p in $PESTS; do
  s="$(date -Is)"
  echo; echo "############ server$SERVER -> $p ############"
  if bash "$AP/run_pest.sh" "$p"; then st=OK; else st=FAILED; rc_all=1; fi
  printf '%s\t%s\t%s\t%s\n' "$p" "$st" "$s" "$(date -Is)" >> "$SUMMARY"
  echo "[server$SERVER] $p -> $st"
done

echo; echo "===== server$SERVER summary ====="; column -t "$SUMMARY"
exit $rc_all
