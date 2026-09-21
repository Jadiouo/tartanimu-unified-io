#!/bin/bash
# One-shot bootstrap for a fresh GPUtw instance: bundle from /vault, every patch from
# /vault/tartanimu/patch, val solution, caches directory; then start the lane runner
# in wait mode on runs/lanes/<LANE>.list (lines can be appended while it runs).
# Usage: bootstrap_full.sh <LANE_NAME>
set -e
LANE=${1:-Q}
cd /workspace
[ -d TartanIMU ] || { tar -xf /vault/tartanimu/bundle.tar && mv gputw_bundle TartanIMU; }
cd TartanIMU
P=/vault/tartanimu/patch
cp $P/lane2.sh runs/lane2.sh
for f in updir.py dense.py train_v2.py pretrain_dense.py fine_context.py moe.py calnet.py attnet.py gtup.py; do [ -f $P/$f ] && cp $P/$f unified/; done
for f in synth_flights_2026_09_16.py tta_time.py; do [ -f $P/$f ] && cp $P/$f local_eval/; done
mkdir -p runs/stress_fold && cp $P/stress_fold.json runs/ && cp $P/stress_fold_b.json runs/ 2>/dev/null; cp $P/drone_stats_2026-09-17.csv runs/stress_fold/
[ -f /vault/tartanimu/dense600_s42.pt ] && cp /vault/tartanimu/dense600_s42.pt runs/ssl/
mkdir -p runs/attnet runs/calnet; for f in attnet_final.pt calnet_final.pt; do [ -f /vault/results/INS/models/$f ] && cp /vault/results/INS/models/$f runs/${f%%_*}/; done; for f in stress_fold_sprint.json stress_fold_sprint2.json stress_fold_sprint3.json; do [ -f $P/$f ] && cp $P/$f runs/; done
python3 -c "import torch, numpy, pandas, scipy" 2>/dev/null || pip install -q numpy pandas scipy
[ -s local_eval/val_solution.csv ] || python3 local_eval/build_val_solution.py > /dev/null
mkdir -p /workspace/cache runs/lanes/logs /vault/results/$LANE
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
grep -c up_for_files unified/updir.py; grep -c TARTANIMU_CUDNN unified/train_v2.py; grep -c 'exec 9>&-' runs/lane2.sh; grep -c drag_head unified/train_v2.py
touch runs/lanes/$LANE.list
export LANE_CMD="python3 unified/train_v2.py" GPU_SLOTS=1 LANE_WAIT=1 LANE_NOSHELL=1 MAXRUN_MIN=100
export TARTANIMU_CACHE=/workspace/cache TARTANIMU_CUDNN_BENCHMARK=0
# result saver: every finished run is tarred into /vault/results/<LANE>/ (polls the lane log)
nohup bash -c "cd /workspace/TartanIMU; while true; do for t in \$(grep -oE '^--- [A-Za-z0-9_]+ done' runs/lanes/$LANE.nohup 2>/dev/null | awk '{print \$2}'); do [ -f /vault/results/$LANE/\$t.tgz ] || { tar -czf /vault/results/$LANE/\$t.tgz unified/history_\$t.csv unified/ckpt_\${t}_last.pt local_eval/sub_val_\${t}_last.csv local_eval/sub_hold_\${t}_last.csv local_eval/sub_test_\$t.csv runs/lanes/logs/\$t.log 2>/dev/null; echo \"\$(date +%T) saved \$t\" >> /vault/results/$LANE/saver.log; }; done; sleep 60; done" > /dev/null 2>&1 &
nohup bash runs/lane2.sh $LANE > runs/lanes/$LANE.nohup 2>&1 &
echo BOOTSTRAP_OK
