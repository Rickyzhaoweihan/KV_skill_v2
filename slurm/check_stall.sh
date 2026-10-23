#!/bin/bash
# Flags wasted training: runs whose held-out val never beats step 0, and
# runs that select/evaluate the step-0 checkpoint. Prints one line per problem.
cd "$(dirname "$0")/.." || exit 1
for f in $(ls -t slurm/logs/D*.out 2>/dev/null | head -40); do
  n=$(grep -E "\[val @ [1-9]" "$f" | grep -cE "n=[0-9]{2,}")
  if [ "$n" -ge 3 ]; then
    if ! grep -E "\[val @ [1-9][0-9]*\]" "$f" | grep -E "n=[0-9]{2,}" | grep -q "\*best\*"; then
      echo "STALL $f: $n val checks after step 0, none beat step 0"
    fi
  fi
  grep -H -E "best val crit .* \(step 0\)|saved best at step 0" "$f" 2>/dev/null | sed 's/^/STEP0 /'
  # trained-layer gates all exactly 0 at eval time => new layers inert
  grep -H -E "^D13 eval:.*gates \[.*0\.0, 0\.0, 0\.0, 0\.0\]" "$f" 2>/dev/null | cut -c1-160 | sed 's/^/INERT /'
done
