#!/bin/bash
# exec a shell command in an instance: gputw_exec.sh <id> <timeout_ms> "<sh -c command>"
set -a; source ~/.config/gputw/env; set +a
python3 - "$1" "$2" "$3" <<'PY'
import json, sys, os, urllib.request
iid, tmo, cmd = sys.argv[1], int(sys.argv[2]), sys.argv[3]
req = urllib.request.Request(f"https://api.gputw.ai/api/instances/{iid}/exec", data=json.dumps({"command": ["sh", "-c", cmd], "timeoutMs": tmo}).encode(),
    headers={"Authorization": "Bearer " + os.environ["GPUTW_API_KEY"], "User-Agent": "TartanIMU-cc/1.0", "Content-Type": "application/json"}, method="POST")
try:
    d = json.load(urllib.request.urlopen(req, timeout=tmo // 1000 + 30))
except Exception as e:
    print("HTTP error:", e); sys.exit(2)
if not d.get("success"): print("API error:", d.get("error")); sys.exit(1)
x = d["data"]; sys.stdout.write(x.get("stdout", "")); sys.stderr.write(x.get("stderr", "")); print(f"[exit {x.get('exitCode')}]"); sys.exit(0)
PY
