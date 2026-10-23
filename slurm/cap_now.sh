#!/bin/bash
# Prints the current job cap for rickyhan (running + pending), set by how busy the lab GPU partitions are.
# User policy (2026-10-19): if many other users have jobs pending on the lab partitions, cap at 5 to be polite;
# 8 when a few other jobs are waiting, 14 when nobody else is waiting for GPUs (raised from 10 by the user, 2026-10-21).
# Output: "<cap> <reason>"
PARTS="drjieliu-h100|drjieliu-l40s"
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
