#!/bin/bash
# Poll the vault for finished cards of the given lanes, pull each new <tag>.tgz once, run the
# matching ruler (sprint folds 1/2/3, fold A) or print the last epoch line (twins / train+val),
# append everything to runs/collect_<date>.log. usage: cloud/collect_lanes.sh <hours> <lane>...
set -a; source ~/.config/gputw/env; set +a; cd "$(dirname "$0")/.."
end=$(( $(date +%s) + ${1:-6}*3600 )); shift; LANES="$@"; LOG=runs/collect_$(date +%m%d).log; mkdir -p runs/gputw_results/x
while [ $(date +%s) -lt $end ]; do
  for L in $LANES; do
    for t in $(grep -oE '^[A-Za-z0-9_]+' runs/lanes/$L.list 2>/dev/null); do
      [ -f runs/gputw_results/x/$t/.done ] && continue
      code=$(curl -s -o runs/gputw_results/$t.tgz -w "%{http_code}" "https://upload.gputw.ai/api/vault/download?filename=results/$L/$t.tgz" -H "Authorization: Bearer $GPUTW_API_KEY" -H "User-Agent: cc")
      [ "$code" = 200 ] || { rm -f runs/gputw_results/$t.tgz; continue; }
      mkdir -p runs/gputw_results/x/$t && tar -xzf runs/gputw_results/$t.tgz -C runs/gputw_results/x/$t 2>/dev/null || { echo "$(date +%H:%M) $t: bad tgz" >> $LOG; continue; }
      touch runs/gputw_results/x/$t/.done
      echo "== $(date +%H:%M) $t ($L)" >> $LOG
      grep -E "^ep +120 |^ep +[0-9]+ .*\*$" runs/gputw_results/x/$t/runs/lanes/logs/$t.log 2>/dev/null | tail -1 | cut -c1-400 >> $LOG
      case $t in
        sp1_*) fold=runs/stress_fold_sprint.json;; sp2_*) fold=runs/stress_fold_sprint2.json;; sp3_*) fold=runs/stress_fold_sprint3.json;; fa_*) fold=runs/stress_fold.json;; *) fold="";;
      esac
      if [ -n "$fold" ] && [ -f runs/gputw_results/x/$t/local_eval/sub_hold_${t}_last.csv ]; then
        cp runs/gputw_results/x/$t/local_eval/sub_hold_${t}_last.csv local_eval/
        .venv/bin/python local_eval/sprint_check.py --fold $fold --tag $t 2>&1 | grep -v Warn | tail -1 >> $LOG
      fi
    done
  done
  sleep 120
done
echo "collector end $(date +%H:%M)" >> $LOG
