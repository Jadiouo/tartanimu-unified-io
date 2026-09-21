#!/bin/bash
# Restart a stopped instance (same node, else the cheapest alternative), bootstrap lane <LANE>
# from /vault (bundle + patch), queue runs/lanes/<LANE>.list (already uploaded to the patch dir)
# and arm the lane self-stop. usage: cloud/launch_lane.sh <idfile e.g. gputw_M2> <LANE> <CAP_H>
set -a; source ~/.config/gputw/env; set +a; cd "$(dirname "$0")/.."
f=$1; LANE=$2; CAP=${3:-5}; id=$(cat runs/lanes/$f.id); H=(-H "Authorization: Bearer $GPUTW_API_KEY" -H "User-Agent: cc" -H "Content-Type: application/json")
st=$(curl -s "https://api.gputw.ai/api/instances/$id" "${H[@]}" | grep -oE '"status":"[A-Z_]+"' | head -1)
if ! echo "$st" | grep -q RUNNING; then
  r=$(curl -s -X POST "https://api.gputw.ai/api/instances/$id/restart" "${H[@]}" -d '{"mode":"same"}')
  if echo "$r" | grep -q '"success":false'; then
    NODE=$(curl -s "https://api.gputw.ai/api/instances/$id/restart-options" "${H[@]}" | python3 -c "import json,sys; a=[x.get('node',x) for x in json.load(sys.stdin)['data']['alternatives']]; a.sort(key=lambda n: n.get('hourlyRate',9)); print(a[0]['id'])")
    r=$(curl -s -X POST "https://api.gputw.ai/api/instances/$id/restart" "${H[@]}" -d "{\"mode\":\"alternative\",\"nodeId\":\"$NODE\"}")
    new=$(echo "$r" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['id'] if d.get('data') else '')" 2>/dev/null)
    [ -n "$new" ] && { cp runs/lanes/$f.id runs/lanes/${f}_old_$(date +%H%M).id; echo "$new" > runs/lanes/$f.id; id=$new; }
  fi
  echo "$f restart: $(echo "$r" | grep -oE '"(status|error)":"?[^",}]+' | head -1) id=$id"
  for i in $(seq 1 60); do curl -s "https://api.gputw.ai/api/instances/$id" "${H[@]}" | grep -qE '"status":"RUNNING"' && break; sleep 10; done
fi
bash cloud/gputw_exec.sh $id 30000 "cd /workspace && nohup bash /vault/tartanimu/patch/bootstrap_full.sh $LANE > /workspace/bootstrap.log 2>&1 & echo started" | head -1
for i in $(seq 1 90); do out=$(bash cloud/gputw_exec.sh $id 30000 "grep -E 'BOOTSTRAP_OK|rror' /workspace/bootstrap.log 2>/dev/null | head -1" 2>/dev/null | grep -v '^\[exit' | head -1); [ -n "$out" ] && break; sleep 10; done
echo "$f bootstrap: $out"
bash cloud/gputw_exec.sh $id 60000 "cd /workspace/TartanIMU && cat /vault/tartanimu/patch/$LANE.list >> runs/lanes/$LANE.list && wc -l < runs/lanes/$LANE.list && ls runs/attnet runs/calnet runs/stress_fold_sprint2.json | tr '\n' ' ' ; GPUTW_API_KEY=$GPUTW_API_KEY nohup bash /vault/tartanimu/patch/ss_lane.sh $id $LANE $CAP >/dev/null 2>&1 & sleep 1; pgrep -f 'ss_lan[e]' | wc -l" | tr '\n' ' '; echo
