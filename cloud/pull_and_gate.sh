#!/bin/bash
# S-fast gates: pull <lane>/<tag>.tgz from the Vault, place hold CSV + ckpt, run ruler 1 and ruler 2.
# usage: cloud/pull_and_gate.sh <lane> <tag> [base tags for sprint_check...]
set -e; cd "$(dirname "$0")/.."; set -a; source ~/.config/gputw/env; set +a
lane=$1; tag=$2; shift 2
mkdir -p runs/gputw_results/x/$tag
curl -s -H "Authorization: Bearer $GPUTW_API_KEY" -H "User-Agent: cc" -o runs/gputw_results/$tag.tgz "https://upload.gputw.ai/api/vault/download?filename=results/$lane/$tag.tgz"
tar -xzf runs/gputw_results/$tag.tgz -C runs/gputw_results/x/$tag
cp runs/gputw_results/x/$tag/local_eval/sub_hold_${tag}_last.csv local_eval/
args=""; for b in "$@"; do args="$args --tag $b"; done
.venv/bin/python local_eval/sprint_check.py $args --tag $tag 2>&1 | grep -v Warn
CUDA_VISIBLE_DEVICES="" .venv/bin/python local_eval/drag_sensitivity.py --weights runs/gputw_results/x/$tag/unified/ckpt_${tag}_last.pt 2>&1 | grep -v Warn
