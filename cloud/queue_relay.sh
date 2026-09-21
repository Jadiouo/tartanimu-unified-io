#!/bin/bash
# Helper-instance relay (learned-INS pipeline, 2026-09-19): every 30 s append new lines from
# /vault/results/INS/<LANE>_queue.list to the local lane list, and sync sub-module weights
# from /vault/results/INS/models/ into runs/attnet, runs/calnet. Stops on <LANE>_queue.END.
# usage: nohup bash cloud/queue_relay.sh Q1 &
LANE=${1:-Q1}; cd /workspace/TartanIMU; Q=/vault/results/INS/${LANE}_queue.list; n=0
mkdir -p runs/attnet runs/calnet /vault/results/INS
while true; do
  cp -n /vault/results/INS/models/*.pt runs/attnet/ 2>/dev/null; cp -n /vault/results/INS/models/*.pt runs/calnet/ 2>/dev/null
  if [ -f "$Q" ]; then
    total=$(grep -c . "$Q"); if [ "$total" -gt "$n" ]; then sed -n "$((n+1)),${total}p" "$Q" >> runs/lanes/$LANE.list; n=$total; echo "$(date +%T) relayed to $n" >> /vault/results/INS/relay_$LANE.log; fi
  fi
  [ -f /vault/results/INS/${LANE}_queue.END ] && { echo END >> runs/lanes/$LANE.list; echo "$(date +%T) END" >> /vault/results/INS/relay_$LANE.log; break; }
  sleep 30
done
