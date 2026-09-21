#!/bin/bash
# Learned-INS pipeline orchestrator (docs/schedules/learned_ins_build_spec_2026-09-18.md),
# self-driving on the main instance (5090, lane Q2) with the helper instance (3090, lane Q1)
# fed through /vault/results/INS/Q1_queue.list (cloud/queue_relay.sh). Every card, gate and
# decision is logged to /vault/results/INS/pipeline.log; artefacts stay in /vault/results/<lane>/.
# Nothing is submitted from here (Kaggle / official queries need the local machine).
# usage (on the 5090): nohup bash cloud/ins_pipeline.sh > /vault/results/INS/pipeline.nohup 2>&1 &
set -u; cd /workspace/TartanIMU
R=/vault/results; INS=$R/INS; mkdir -p $INS/models; L=$INS/pipeline.log
log(){ echo "$(date +%F\ %T) $*" | tee -a $L; }
V6="--huber_beta 0.05 --seg_len 40 --init_trunk runs/ssl/dense180_tv_s42.pt --wide --yaw_ident 0.5 --yaw_dironly --save_last"
SF="--tdil 0.7 1.5 --tdil_fast 0.7 1.2 --sdil_fast 1.0 1.8 --aug_mix"
SPR="--trainval --holdout runs/stress_fold_sprint.json"; FA="--trainval --holdout runs/stress_fold.json"
queue(){ lane=$1; shift; line="$*"; if [ "$lane" = Q2 ]; then echo "$line" >> runs/lanes/Q2.list; else echo "$line" >> $INS/Q1_queue.list; fi; log "queued [$lane] ${line%% *}"; }
wait_done(){ lane=$1; tag=$2; t0=$(date +%s); while [ ! -f $R/$lane/$tag.tgz ]; do
    if grep -aq "FAILED $tag" runs/lanes/Q2.nohup 2>/dev/null; then log "FAILED $tag (Q2)"; return 1; fi
    if [ $(( $(date +%s) - t0 )) -gt 14400 ]; then log "TIMEOUT waiting for $tag ($lane) after 4 h"; return 1; fi
    sleep 60; done; log "done $tag ($lane)"; return 0; }
fetch(){ lane=$1; tag=$2; mkdir -p /tmp/x/$tag; tar -xzf $R/$lane/$tag.tgz -C /tmp/x/$tag 2>/dev/null
  cp /tmp/x/$tag/local_eval/sub_hold_${tag}_last.csv local_eval/ 2>/dev/null; cp /tmp/x/$tag/unified/ckpt_${tag}_last.pt unified/ 2>/dev/null; cp /tmp/x/$tag/unified/history_$tag.csv unified/ 2>/dev/null; }
sprint(){ python3 local_eval/sprint_check.py --fold runs/stress_fold_sprint.json "$@" 2>/dev/null | grep -v Warn | tee -a $L; }
sprint_vals(){ python3 local_eval/sprint_check.py --fold runs/stress_fold_sprint.json --tag $1 2>/dev/null | awk -v t=$1 '$1==t{print $4, $6}'; }   # sprint_mean pred/true
sens(){ CUDA_VISIBLE_DEVICES='' python3 local_eval/drag_sensitivity.py --weights unified/ckpt_$1_last.pt 2>/dev/null | grep -v Warn | tee -a $L; }
valrow(){ tail -1 unified/history_$1.csv | awk -F, -v t=$1 '{printf "%s val %.4f car %.4f dog %.4f drone %.4f human %.4f hold %.3f\n",t,$2,$13,$14,$15,$16,$17}' | tee -a $L; }
export TARTANIMU_CUDNN_BENCHMARK=0

log "=== learned-INS pipeline start ==="
cp $INS/hold/*.csv local_eval/ 2>/dev/null     # reference hold CSVs (sp_base, sp_sfast, sf_sfast18) for sprint_check
# ---------- A: cards already queued (sp_dr_s42 on Q2; sp_dr_s43, sp_drnofuse_s42 on Q1) ----------
wait_done Q2 a3v16S_s42; fetch Q2 a3v16S_s42; valrow a3v16S_s42; sens a3v16S_s42
wait_done Q2 sp_dr_s42; wait_done Q1 sp_dr_s43; wait_done Q1 sp_drnofuse_s42
for t in sp_dr_s42 sp_dr_s43 sp_drnofuse_s42; do lane=Q2; [ -f $R/Q1/$t.tgz ] && lane=Q1; fetch $lane $t; done
sprint --tag sp_base_s42 --tag sp_sfast_s42 --tag sp_dr_s42 --tag sp_dr_s43 --tag sp_drnofuse_s42
for t in sp_dr_s42 sp_dr_s43 sp_drnofuse_s42; do sens $t; done
read m42 p42 <<< "$(sprint_vals sp_dr_s42)"; read m43 p43 <<< "$(sprint_vals sp_dr_s43)"
A_PASS=$(python3 -c "print(int((($m42+$m43)/2 <= 1.7) and (($p42+$p43)/2 >= 0.88)))")
log "GATE A: sp_dr mean sprint $(python3 -c "print(round(($m42+$m43)/2,2))") pred/true $(python3 -c "print(round(($p42+$p43)/2,2))") -> pass=$A_PASS (gate <=1.7, >=0.88; sp_sfast 2.10/0.81)"
if [ "$A_PASS" = 1 ]; then
  queue Q1 "tv_dr_s42 --seed 42 --data_seed 42 --grav slowdr --fuse $V6 --init_trunk_pad $SF"
  queue Q2 "a3v17_s42 --seed 42 --data_seed 42 --grav slowdr --fuse $V6 --init_trunk_pad $SF --trainval"
  queue Q1 "a3v17_s43 --seed 43 --data_seed 43 --grav slowdr --fuse $V6 --init_trunk_pad $SF --trainval"
fi

# ---------- C then B: final CalNet + AttNet on train+val (weights shared through the vault) ----------
log "training CalNet final (train+val)"; python3 unified/train_calnet.py --tag final --trainval --steps 3000 2>&1 | grep -v Warn | tail -2 | tee -a $L
cp runs/calnet/calnet_final.pt $INS/models/
log "training AttNet final (train+val, CalNet-calibrated)"; python3 unified/train_attnet.py --tag final --trainval --calnet runs/calnet/calnet_final.pt --steps 2000 2>&1 | grep -v Warn | tail -2 | tee -a $L
cp runs/attnet/attnet_final.pt $INS/models/
log "AttNet gate (held-out flights are IN train+val here: informational only)"
CUDA_VISIBLE_DEVICES='' python3 local_eval/attnet_check.py --weights runs/attnet/attnet_final.pt --calnet runs/calnet/calnet_final.pt 2>/dev/null | grep -v Warn | tee -a $L
CUDA_VISIBLE_DEVICES='' python3 local_eval/calnet_check.py --weights runs/calnet/calnet_final.pt 2>/dev/null | grep -v Warn | tee -a $L
sleep 90   # let the Q1 relay pick the weights up
AC="--att runs/attnet/attnet_final.pt --cal runs/calnet/calnet_final.pt"
queue Q2 "sp_cal_s42 --seed 42 --data_seed 42 --grav slowdr --fuse $AC $V6 --init_trunk_pad $SF $SPR"
queue Q1 "sp_cal_s43 --seed 43 --data_seed 43 --grav slowdr --fuse $AC $V6 --init_trunk_pad $SF $SPR"
queue Q1 "fa_cal_s42 --seed 42 --data_seed 42 --grav slowdr --fuse $AC $V6 --init_trunk_pad $SF $FA"
if [ "$A_PASS" = 1 ]; then wait_done Q2 a3v17_s42; fetch Q2 a3v17_s42; sens a3v17_s42; fi
wait_done Q2 sp_cal_s42; wait_done Q1 sp_cal_s43
for t in sp_cal_s42 sp_cal_s43; do lane=Q2; [ -f $R/Q1/$t.tgz ] && lane=Q1; fetch $lane $t; done
sprint --tag sp_base_s42 --tag sp_sfast_s42 --tag sp_dr_s42 --tag sp_cal_s42 --tag sp_cal_s43
for t in sp_cal_s42 sp_cal_s43; do sens $t; done
read m42 p42 <<< "$(sprint_vals sp_cal_s42)"; read m43 p43 <<< "$(sprint_vals sp_cal_s43)"
C_PASS=$(python3 -c "print(int((($m42+$m43)/2 <= 1.5) and (($p42+$p43)/2 >= 0.88)))")
log "GATE C: sp_cal mean sprint $(python3 -c "print(round(($m42+$m43)/2,2))") pred/true $(python3 -c "print(round(($p42+$p43)/2,2))") -> pass=$C_PASS (gate <=1.5, >=0.88)"
# candidates are trained regardless (the morning decision needs them); the gate is recorded
queue Q2 "a3v19_s42 --seed 42 --data_seed 42 --grav slowdr --fuse $AC $V6 --init_trunk_pad $SF --trainval"
queue Q1 "tv_cal_s42 --seed 42 --data_seed 42 --grav slowdr --fuse $AC $V6 --init_trunk_pad $SF"
queue Q1 "a3v19_s43 --seed 43 --data_seed 43 --grav slowdr --fuse $AC $V6 --init_trunk_pad $SF --trainval"
if [ "$A_PASS" = 1 ]; then wait_done Q1 tv_dr_s42; fetch Q1 tv_dr_s42; valrow tv_dr_s42; wait_done Q1 a3v17_s43; fi
wait_done Q1 fa_cal_s42; fetch Q1 fa_cal_s42
python3 local_eval/sprint_check.py --fold runs/stress_fold.json --tag sf_sfast18_s42 --tag fa_cal_s42 2>/dev/null | grep -v Warn | tee -a $L
python3 local_eval/stress_bins.py --tag fa_cal_s42 2>/dev/null | grep -A4 "by recording class" | tee -a $L
wait_done Q2 a3v19_s42; fetch Q2 a3v19_s42; sens a3v19_s42
wait_done Q1 tv_cal_s42; fetch Q1 tv_cal_s42; valrow tv_cal_s42
wait_done Q1 a3v19_s43; fetch Q1 a3v19_s43; sens a3v19_s43
touch $INS/Q1_queue.END; echo END >> runs/lanes/Q2.list
log "=== pipeline complete: A_PASS=$A_PASS C_PASS=$C_PASS; candidates a3v16S_s42 (safety), $( [ "$A_PASS" = 1 ] && echo a3v17_s42/s43,) a3v19_s42/s43 in $R/Q2 and $R/Q1 ==="
touch $INS/PIPELINE.DONE
