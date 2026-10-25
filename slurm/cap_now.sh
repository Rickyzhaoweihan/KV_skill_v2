#!/bin/bash
# Prints the current job cap for rickyhan (running + pending), set by how busy the lab GPU partitions are.
# User policy (2026-10-19): if many other users have jobs pending on the lab partitions, cap at 5 to be polite;
# 8 when a few other jobs are waiting, 14 when nobody else is waiting for GPUs (raised from 10 by the user, 2026-10-21).
# Output: "<cap> <reason>"
PARTS="drjieliu-h100|drjieliu-l40s"
# User override (2026-10-24: "多排点我需要抢卡来加速进程"): if slurm/cap_override holds a number, it is the cap.
OV="$(dirname "$0")/cap_override"; if [ -s "$OV" ]; then
  # User 2026-10-25: cap_override = max NON-HELD KV-Skill GPU jobs (running + pending). Jobs held long-term (not listed in
  # slurm/hold_queue.txt) are added on top so they do not eat slots; jobs in hold_queue.txt (held only to respect the cap,
  # released by the coordinator's releaser in priority order) are NOT added, so waiters cannot overtake them.
  HQ="$(dirname "$0")/hold_queue.txt"; touch "$HQ"
  lt=$(squeue -u rickyhan -h -t PD -o "%i %j %b %r" 2>/dev/null | awk '$2 ~ /^D[0-9]/ && $3 ~ /gpu/ && $4 ~ /Held/ {print $1}' | grep -vxF -f "$HQ" | grep -c .)
  echo "$(( $(tr -dc 0-9 < "$OV") + lt )) override: user (slurm/cap_override non-held + $lt long-term held)"; exit 0; fi
# only jobs actually waiting for GPUs (skip BeginTime / Dependency / held jobs)
others=$(squeue -h -t PENDING -o "%u %P %b %r" 2>/dev/null | awk -v p="$PARTS" '$1!="rickyhan" && $2 ~ p && $3 ~ /gpu/ && $4 !~ /BeginTime|Dependency|Held/')
n_pend=$(echo "$others" | grep -c .)
n_users=$(echo "$others" | awk 'NF{print $1}' | sort -u | grep -c .)
free=0
for n in lh0303 lh0304; do
  line=$(scontrol show node $n 2>/dev/null)
  tot=$(echo "$line" | grep -o 'CfgTRES=[^ ]*' | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2)
  use=$(echo "$line" | grep -o 'AllocTRES=[^ ]*' | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2)
  free=$((free + ${tot:-0} - ${use:-0}))
done
if [ "$n_pend" -ge 4 ] || [ "$n_users" -ge 3 ]; then
  echo "5 busy: others pending=$n_pend users=$n_users free_gpus=$free"
elif [ "$n_pend" -eq 0 ]; then
  echo "14 idle: others pending=0 free_gpus=$free"
else
  echo "8 normal: others pending=$n_pend users=$n_users free_gpus=$free"
fi
