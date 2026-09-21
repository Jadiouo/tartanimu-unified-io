#!/bin/bash
# On-instance self-stop for one lane: stop this instance once every tag in runs/lanes/<LANE>.list
# (re-read each poll, so appended cards count) has its tgz in /vault/results/<LANE>/ or a
# FAILED/REFUSED line in the lane log, or after CAP_H hours. Key comes from the environment only.
# usage: GPUTW_API_KEY=... nohup bash cloud/ss_lane.sh <instance_id> <LANE> <CAP_H> &
id=$1; LANE=$2; end=$(( $(date +%s) + ${3:-4}*3600 )); L=/vault/results/$LANE/self_stop.log; cd /workspace/TartanIMU
alldone(){ n=0; for t in $(awk '{print $1}' runs/lanes/$LANE.list); do n=$((n+1)); [ -f /vault/results/$LANE/$t.tgz ] || grep -aqE "(FAILED|REFUSED) $t( |$)" runs/lanes/$LANE.nohup 2>/dev/null || return 1; done; [ $n -gt 0 ]; }
sleep 300                                              # let the first card start before judging an empty list
while ! alldone && [ $(date +%s) -lt $end ]; do sleep 90; done
sync; sleep 45; echo "$(date +%T) stopping: $(alldone && echo all-done || echo budget)" >> $L
curl -s -X POST "https://api.gputw.ai/api/instances/stop" -H "Authorization: Bearer $GPUTW_API_KEY" -H "User-Agent: cc" -H "Content-Type: application/json" -d "{\"instanceId\":\"$id\"}" | grep -oE '"status":"[A-Z]+"' >> $L
