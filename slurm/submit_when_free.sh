#!/bin/bash
# usage: submit_when_free.sh <max_jobs|auto> <env assignments...> -- sbatch args...
# The cap is DYNAMIC (slurm/cap_now.sh: 5 when other users queue up on the lab partitions, 8 normally,
# 10 when idle). Counts only KV-Skill (D*) GPU jobs — user, 2026-10-20: other projects (e.g. hi_ko) do not block KV-Skill. A numeric first argument is kept for old callers but only lowers the cap, never raises it.
# PRIORITY (user, 2026-10-21): waiters register a ticket in slurm/waiters/; when a slot frees, only the waiter with
# the best rank (slurm/priority.txt, then oldest ticket) submits. Lower-priority waiters keep waiting.
ARG=$1; shift
ENVS=(); while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
HERE=$(cd "$(dirname "$0")" && pwd)
# CPU-only jobs (slurm/cpu.sbatch, no GPU) never wait on the GPU cap.
case " $* " in *cpu.sbatch\ *) exec env "${ENVS[@]}" sbatch --parsable "$@";; esac
NAME=$(printf '%s\n' "$@" | sed -n 's/^--job-name=//p' | head -1)
[ -z "$NAME" ] && NAME=$(printf '%s\n' "$@" | grep -A1 -x -- '--job-name' | tail -1)
[ -z "$NAME" ] && NAME=unnamed
rank_of() { local n=$1 r=0 line; while IFS= read -r line; do case "$line" in ''|\#*) continue;; esac; r=$((r+1)); echo "$n" | grep -qE "$line" && { echo $r; return; }; done < "$HERE/priority.txt"; echo 99; }
# H100-only jobs (27B) compete only with other H100-only waiters, so a free H100 never idles behind L40S jobs.
case " $* " in *drjieliu-l40s*) PART=l40s;; *drjieliu-h100*) PART=h100;; *) PART=l40s;; esac
W="$HERE/waiters"; mkdir -p "$W"
T="$W/$(date +%s).$$.$PART.$NAME"; touch "$T"; trap 'rm -f "$T"' EXIT
mine=$(basename "$T")
while true; do
  CAP=$(bash "$HERE/cap_now.sh" | cut -d' ' -f1)
  case "$ARG" in ''|auto|*[!0-9]*) ;; *) [ "$ARG" -lt "$CAP" ] && CAP=$ARG ;; esac
  if [ "$(squeue -u rickyhan -h -o "%j %b" | grep -E "^D[0-9]" | grep -c gpu)" -lt "$CAP" ]; then
    best=""; bestkey=""
    for f in "$W"/*; do
      [ -e "$f" ] || continue; b=$(basename "$f"); pid=$(echo "$b" | cut -d. -f2)
      kill -0 "$pid" 2>/dev/null || { rm -f "$f"; continue; }
      [ "$(echo "$b" | cut -d. -f3)" = "$PART" ] || continue
      key="$(printf '%02d' "$(rank_of "${b#*.*.*.}")").$b"
      { [ -z "$bestkey" ] || [[ "$key" < "$bestkey" ]]; } && { bestkey=$key; best=$b; }
    done
    [ "$best" = "$mine" ] && break
  fi
  sleep 60
done
env "${ENVS[@]}" sbatch --parsable "$@"
