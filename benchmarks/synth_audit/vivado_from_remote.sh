#!/bin/bash
# Second stage of the audit when the first one runs on another machine (no Vivado licence there):
# fetch the finished cases, run Vivado on them here, until every case has its Vivado result.
#   [JOBS=2] ./vivado_from_remote.sh user@host:/path/to/audit_out LOCAL_OUT_DIR
REMOTE=$1; OUT=$2
HOST=${REMOTE%%:*}; RDIR=${REMOTE#*:}
mkdir -p "$OUT"
TOTAL=$(python3 "$(dirname "$0")/audit.py" list | wc -l)
while true; do
  for case in $(ssh -o BatchMode=yes "$HOST" "cd $RDIR 2>/dev/null && ls */result.json 2>/dev/null | xargs -r -n1 dirname"); do
    [ -e "$OUT/$case/result.json" ] || rsync -a "$HOST:$RDIR/$case" "$OUT/"
  done
  # JOBS Vivado runs side by side (default 1), each on its own share of the cases
  JOBS=${JOBS:-1}
  for i in $(seq 0 $((JOBS - 1))); do
    AUDIT_NO_WAIT=$([ "$JOBS" -gt 1 ] && echo 1) python3 "$(dirname "$0")/audit.py" run "$OUT" --stage vivado --shard $i/$JOBS &
  done
  wait
  done_cases=$(grep -l '"vivado"\|"error"' "$OUT"/*/result.json 2>/dev/null | wc -l)
  echo "--- $done_cases of $TOTAL cases have a Vivado result or an error"
  [ "$done_cases" -ge "$TOTAL" ] && break
  sleep 60
done
