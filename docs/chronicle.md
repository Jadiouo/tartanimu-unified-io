# TartanIMU Challenge — chronicle (2026-09-08 → 09-21)

A day-by-day record of the 13 days: what was tried, what it scored on which ruler, what was decided, and what the artefacts are. Times are Taipei (GMT+8). Every number carries a tag that can be traced to `runs/official/` (official scoring service) or to the training run of that name.

---

## 0. Conventions

**Five rulers** (lower is better; only compare within one ruler)
| Ruler | Definition | Population it measures | Noise | In use from |
|---|---|---|---|---|
| **val** | released val (80 recordings) scored with the official scorer, train-only models | drone group D + car/dog/human (in-distribution) | seed sd ≈ 0.002 | 9/8 |
| **public LB** | Kaggle 30 % | skewed to fast drone flights | — | 9/8 |
| **official full test** (scoring service) | 89 test recordings, per-platform ATE20/AVE/RTE + per-sequence | everything (drone 68–72 % of the score) | same-recipe seed sd ≈ 0.005 | 9/12 |
| **hold-out** → **fold A** (`runs/folds/stress_fold.json`, 26 recordings) | 12 original hold-outs + 5 high-rotation + 7 fast + 2 middle, both IMUs of a flight held out together; no training window above 8 m/s | group C + pure extrapolation | sd 0.02–0.04, machine-to-machine 0.06 | 9/17 |
| **sprint folds** (`stress_fold_sprint*.json`, folds 1/2/3) | hold out one outdoor sprint pair (0037/39, 0038/40, 0041/42) plus the aggressive pair 0021/22; the other sprints stay in training | groups A/B (the tail) | sd ≈ 0.3 (two sequences) | 9/18 night / 9/20 night |

**Four groups of test drone sequences** (defined 9/17 from the official per-sequence table; Seq ID = test file index + 1): A sprint tail #29/#56 (test_0028/0055, 17 windows, one flight with two IMUs); B secondary tail #37/#38 (42 windows); C eleven high-rotation sequences (mean gyro > 1.5 rad/s); D the remaining thirty. A+B carry 43 % of drone AVE, C 23 %, D 34 %. val only sees D.

**Tag naming**: `a3v<N>_s<seed>` = train+val final candidate; `tv_*` = train-only clean-val twin; `sf_*/fa_*` = fold A; `sp_*/sp1|2|3_*` = sprint folds; `d16_*/d17_*` = 9/16–17 train-only cards; `R3/B0/C0/YA/YB` = 9/15 hold-out cards. Artefacts: `unified/ckpt_<tag>_last.pt`, `local_eval/sub_{val,hold,test}_<tag>*.csv`, `unified/history_<tag>.csv`, `runs/official/<tag>_{overall,platform,sequences}.csv`.

**Score**: Score = 0.6·AVE/0.7356 + 0.4·ATE20/3.1160, each macro-averaged over the four platforms. All-zero = 1.000.

---

## 1. One-page overview

| | |
|---|---|
| Final standing | **20th of 131 teams** (private leaderboard 0.19912, published 2026-09-28) |
| Delivered | **a3v20_s42**: official **0.25878** (ATE20 0.61704 / AVE 0.22017), public 0.32793; HF `LexHo/tartanimu-a3v20` |
| Second entry | a3v21_s42: official 0.26306, public 0.32695; HF `LexHo/tartanimu-a3v21` |
| Start → end | val 1.015 (all-zero) → 0.1746 (tv_film); public 1.054 → 0.3270; official 0.3057 (first query, 9/12) → 0.2588 |
| Competitors | AxisTilted2 official 0.2156, BShankar 0.2379, xunden 0.2585 (single pass); 84 % of our gap is drone AVE, four fifths of it on the 41 non-sprint drone sequences |
| Official queries / Kaggle submissions / training runs | 33 / 26 / ≈ 500+ (local GPU + Kaggle T4 + cloud 3090/4090/5090, ≈ 90 GPU-h) |

**Five turning points**
1. 9/8: the official 4-head baseline is a dead end on the anonymised test → train one unified model; val 0.232 the same night (dense windowing + cross-window GRU).
2. 9/12–13: val plateau (0.21) → masked-IMU self-supervised pre-training of the trunk (−0.014), and the discovery that drone IMU frames differ from the label frame and come from two sources.
3. 9/14: physics-consistent time dilation (T 0.7–1.5) — fastest drone quintile −24 %, official 0.2837 → 0.2687.
4. 9/17–18: new rulers (fold A, sprint fold) that see the tail; the big swings (SSL 3.3×, synthetic flights, JEPA, external NeuroBEM/UZH) all fail or are ruled out; the tail mechanism is found (the models do not read drag) → **S-fast**, official 0.2702 → 0.26577.
5. 9/19–20: learned inertial channels (CalNet/AttNet/v_DR/gate); the first version a3v19 leaks into car; after review fixes (bounded v_DR, 6-D rotation, EKF-style recursion head) → **a3v20 0.25878**; the night of 9/20 closes the recursion line (the gain is one knob with one trade-off); FiLM v1 is seed-robust → a3v21 as the second entry; a3v22 (a3v20 + FiLM) 0.26359 does not replace it.

---

## 2. Day by day (goal → what was done → result → decision → artefacts → lesson)

### 9/8 (Mon) from zero to val 0.232
- **Goal**: data validation, environment, all-zero submission, official baseline.
- **Done**: data complete (test 89 files / 30,644 windows); `.venv`; all-zero public 1.05383 (= by definition); local official scorer pipeline (feeding the ground truth back collapses ATE20 to 0.0128 → sampling convention correct); ran the official `unified.pt` (three repo bugs: hard wandb dependency, `model_param` order, val directory layout).
- **Result**: the 4-head baseline with the best forced head is public 0.937 (all-zero 1.054); `compute_all_heads=False` is a compute shortcut, not routing — the model has no mechanism to infer the platform; the advertised 0.637 cannot be reproduced from the released weights → **the official multi-head model is a dead end on the anonymised test.**
- Same night (three parallel implementations): five ideas, two work — **cross-window GRU context −0.056**, **dense windowing −0.036**; per-frame supervision and gravity auxiliaries are null; scale-aware output worse; Transformer mixer fails. A confound was caught: several trajectories per batch (`batch_segs` 26 → 52 differs by 0.04). Round five saturated three dimensions at once (context length, mixer capacity, training length) → val 0.232 (4 seeds).
- **Decision**: train a unified model from scratch; adopt a paired-bootstrap + multi-seed protocol (threshold ±0.006–0.011). Process lesson: propose and execute separately (a sixth round was started without approval and reverted).
- **Artefacts**: `unified/dense.py` (dense windowing), `ContextIMUNet` (20-window GRU), `report.py`.

### 9/9 (Tue) val selects models; Kaggle compute fails; fine mixer
- Two seeds of one recipe submitted (g20d_s45 val 0.2277 → public 0.40598, rank 22/81; s43 0.2356 → 0.41410): **val gap 0.0079 → public gap 0.0081, ratio 1.03 → val can select models** (same architecture). Linear calibration `public = 0.867·val + 0.217` (R² 0.997); val is optimistic by ~0.18 in absolute terms.
- Kaggle kernels: three failures (paths, search depth, 590 MB re-decoded on cold storage every run, 55 min) → disk cache (6.5 s → 0.4 s); the account has no GPU (phone verification) → shelved.
- Rounds 7–9: **fine mixer fine=5 −0.0121** (five tokens per window into the GRU; 1–25 is a plateau, not a peak); `decompose` (additive segment + deviation) +0.0043 with 70 % from ATE20 → a lever for cross-window correlated error, judged harmful; four MoE placements (Kaggle T4): only FiLM conditioning of the trunk `mf4` wins slightly (−0.0013, within noise). Cross-architecture val→public transfer is 3× steeper than same-architecture. Literature: EqNIO (O(2) equivariance), MosaicIMU, FTIN, DINS-IO.
- Submissions: f5_s42 public 0.39096, dec_s42 0.40612.

### 9/10 (Wed) MoE all null; early-stopping audit; slow gravity channels; physics integration ruled out
- Four MoE placements, 2 seeds, all null (what the rules encourage has no effect on this data). mf4_s42 public 0.39744.
- **Early-stopping audit**: 18 of 125 runs had been stopped early; truncation at 67 % costs 0.010; `--patience` off by default (OneCycle should not be cut).
- **Complementary-filter gravity decomposition**: α sweep — drone wants fast (0.05), the others slow (0.005); only slow works (gsP −0.0053, later −0.0057 with 8 seeds), both together cancel. A vanished cache made runs silently 20× slower → moved to `runs/cache/` + `CACHE_SCHEMA`.
- **Physics integration ceiling**: `dv/dt = f + g − ω×v` with ground-truth gravity and initial velocity: 1 s val 0.659 (learned model 0.219), 5 s 2.23, 20 s 6.96, growing linearly → **sensor error dominates within one second**; the filter's gravity beats the ground-truth gravity (0.539). The premise of "integrate + learn a correction" does not hold.

### 9/11 (Thu) equivariant frame ruled out; external review fixes three bugs; Huber β; round 16
- **EqNIO-style O(2) equivariance** (`unified/equivariant.py`): four symmetry bugs fixed to 1e-15, yet val 0.4428 (f5 0.2188), training loss stuck at 0.25 (27×) → the premise (accurate gravity) fails; closed.
- **External review** found three real bugs: dense sampling crossing trajectories (3.2 %), EMA never evaluated, inference padding treated as observations → fixed (valid lengths, packed GRU, loss masks), 10 unit tests; score effect at noise level.
- Rounds 15/16: continuous 20-s trunk **+0.031** (BatchNorm contamination 0.021 + receptive field 0.014), GradNorm null, **Huber β 0.25 → 0.05 −0.008** (curve 0.2235/0.2178/0.2155/0.2137 bottoms out), **40-window context −0.009** (invisible on the old code because of the boundary bug), `tfilm` gate entropy uniform, `freq` branch negative.
- Rules: ≥ 0.010 to change the base; new conditions screened with one seed first. Flags written: `--continuous/--gradnorm/--huber_beta/--rel_loss/--traj_film/--freq/--norm gn/--short_traj`.

### 9/12 (Fri) all3 fixed; first official decomposition; drone diagnosis
- Three factors combined, **all3 = slow gravity channels + Huber 0.05 + 40 windows**: val 0.2096 (−0.014 vs f5x; pairs do not add, the triple does); public **0.3694, rank 18/81**.
- **First official scoring-service query**: all3_s42 full test 0.30571; car/dog/human are *better* on test than on val, **the whole gap is drone** (test AVE 0.844 = 2.05× val, 72 % of the score); test drone flies harder (acc_dev90 median 1.88 vs val 0.66). Expert table: a drone expert equals the shared model (0.463 vs 0.465) → the drone problem is cross-flight generalisation, not fitting; a car expert −0.009 → sharing costs nothing. A tfilm counterfactual shows the model really uses the whole-trajectory descriptor (car/drone recognise trajectory style). Yaw augmentation fails badly (0.677, training Huber 0.065); 3× oversampling of aggressive drone flights is worse (memorisation).
- Overnight: experts/distillation, SAM, RSC, band, dropout, wd, macro loss, lr, 180 epochs, white noise, no platform CE — all null; time-mask alone −0.005.

### 9/13 (Sat) self-supervised pre-training works (−0.014); the drone-frame discovery; rules check
- **Masked-IMU pre-training** (trunk; reconstruct 60 masked frames × 14 channels): fixed windows 20 epochs −0.010; **dense windows 2 h −0.014** (a3dn 0.1962, 3 seeds sd 0.001); train-only pre-training keeps the gain (optimism from seeing val IMU ≈ 0.002); scale curve monotone but slow (4× steps −0.0016). Every add-on fails: 60 epochs, mask 50 %, segment-level pre-training (trunk+GRU), joint reconstruction fine-tuning (large loss), K=60, time-mask on top. Reading: velocity supervision is 3 numbers per window and the cheapest solution is to memorise trajectory fingerprints; reconstruction gives 280× more targets and puts the trunk in the "IMU structure" basin.
- Official: a3sslc_s42 (pre-training saw val IMU) 0.2962 / public 0.3759 (**public and full test disagree in direction**); a3ssl (with test IMU) 0.2984.
- **Discovery**: drone IMU frames disagree with the label body frame and differ per recording: only 43 of 282 training drone recordings have R_ext ≈ I, the rest are "upside down + some yaw offset" (16 signatures); the noise floor splits drone into **flipped (246/39, median 1.2 m/s, exactly 60 s)** and **identity (43/9, 4.2 m/s, 10× noise floor, ~40 s)** sources; the error sits in the identity group (AVE 0.73 vs 0.33); test estimated 32 flipped / 13 identity, with flights that accelerate longer than any training group. This explains the yaw-augmentation failure, tfilm helping car/drone, expert = memory.
- Rules check: unlabelled-test pre-training judged "allowed with disclosure" that day (reversed on 9/14 after the organisers' reply); TTT conflicts with frozen weights, not done.
- Candidate v1 `a3dnTV_s42` (train+val pre-training 90 min → train+val fine-tune, last epoch) official 0.2837 / public 0.3702.

### 9/14 (Sun) time-dilation augmentation: official 0.2837 → 0.2687; the tail seen for the first time
- **Physics-consistent time dilation T** (`--tdil`): replay drone segments at speed k, resample, ω′ = kω, v′ = kv, **f′ = k²(f − g·up) + g·up** (up from GT, training only). k 1.0–1.3 −0.007 → 0.8–1.3 −0.009 → **0.7–1.5 −0.015** (a3td0715 0.1806; wider to 0.5–2.0 no change); drone only, **fastest quintile −24 to −27 %**; the identity group barely moves (0.731 → 0.708). After fixing "short trajectories never got augmented" the identity group moves −7 % (a3tdA1 0.1802). Translation scaling S 0.8–1.25 is weaker than T and does not add. 240 epochs, augmentation p=0.75, strong k (1.0–2.0 / 1.2–2.5), pre-training on dilated data: all within ±0.003. Capacity closed (width 1.5 loses in three settings).
- Official: **v2 `a3dnTV3td_s42` 0.2687 / public 0.3526 (rank 18)**; v3 (fixed T) 0.2743 (two extreme sequences 4.46 → 5.20, 4.64 → 5.18 vary).
- **Discovery**: test drone error is extremely concentrated — #29 AVE 4.64, #56 4.46, #37 2.99, #38 1.88; the top six are 49 % of drone AVE; our predictions on them cap at 7.9 m/s (= the fastest training trajectory's mean speed); no val trajectory looks like this.
- Organisers' replies (9/5–9/9) collected (`known_problems`): only released train/val may be used; any weights trained on other data (incl. the organisers' checkpoint and self-supervision with test IMU) cannot enter the final; TTA allowed with disclosure; TTT not done.

### 9/15 (Mon) v6 fixed; hold-out; yaw-direction supervision; a dozen rejections
- **Hold-out** (12 drone recordings, 8 flight groups; `runs/folds/holdout.json`) as the tail ruler; R3 = dense120 trunk + T 0.7–1.5 + **`--wide`** (5-s low-pass neighbourhood branch) + holdout: val 0.1808, hold-out 1.147.
- Works: **yaw-direction supervision `--yaw_ident 0.5 --yaw_dironly`** (random rotation of the identity source about the estimated up, direction-only supervision), hold-out 2/2 −0.10 (R3yawd 1.034/1.015); full yaw supervision blows up the speed, p=1.0 is bad.
- Rejected: preint (P1), `--grav adapt`, GT-up diagnostic (attitude hypothesis rejected), data_seed soups collapse (0.3459), C0/C1/C2 (warm-start / wide_to_gru / three-teacher distillation), tdil_short cancel, tdil_fast 0.7–1.0/0.7–2.0 (a3v7/a3v7f/a3v8), YA/YB paired yaw (passes numerically but one pair 0023/0024 contributes 104 %), anti-aliased T (AA; boundary bug fixed, never run locally).
- **v6 = dense180_tv SSL trunk + T 0.7–1.5 + wide + yaw_dironly 0.5 + Huber 0.05 + 40 windows + train+val last epoch**: `a3v6_s42` official **0.2702** / public 0.3566; seeds 43–47 trained (official 9/16: 0.2792/0.2793/0.2809 → family 0.270–0.281). Selection rule **A (seed 42 pre-designated)** adopted.
- Full-history retrospective (reviewer): 329 history files, 425 val CSVs; a GT relative-motion diagnostic (5-s relative rotation/velocity increments + anchor) lowers hold-out AVE 1.14 → 0.56 (an oracle has room), but raw hard reconstruction is 7.6 → no integration branch can be promised.

### 9/16 (Tue) ten cards, zero signal; delivery dry-run v1; big swings at night
- Warm screening + full pairs, ten cards: local cross-attention AT1/AT2, per-window increment-correction head LR0, acc-gain augmentation, context-split, dropout 0.5, 180° yaw, 512-bin expectation head, continuation soups, Lookahead — **all within seed noise (sd 0.002) or worse**. The smoothing family (EMA/SWA/soup/Lookahead) is closed. Time-warp TTA all worse.
- Delivery dry-run v1 (`hf/`, a3v6_s42 sha256 594ffb50…) passes; technical report draft v1.
- Rules snapshot 9/16: external data allowed but must not contain test labels.
- Evening (big swings authorised): S1 SSL scaled 3.3× (234,870 updates, 5 h); **S2 external data UZH-FPV: a leakage check shows UZH-FPV *is* the competition's identity source (13 segments = train/val, 3 = test incl. #29/#56) → its ground truth equals private test labels; banned and deleted**; S3 synthetic aggressive flights (GT → IMU reconstruction, flatness closure, 96 recordings in `data/synth_v1`); TTT diagnostics (AdaBN, masked recon) all negative, closed.

### 9/17 (Wed) big swings all fail; new ruler (fold A); 15 nulls; a3v10
- Cloud overnight (3090 ×2; 4090 ×2 node failures refunded) 15 runs; official 5 queries: train-only base d16_full_base 0.2831; **ssl600 0.2868, ssl600+lr×0.3 0.2895, JEPA 0.2973, synth4 0.3010, all worse**; the tail #29/#56 is 4.8–5.3 on all five. The hold-out's −0.08 was a false signal (rerun variance 0.14).
- **Fold A** (`runs/folds/stress_fold.json`, 26 recordings): brings the tail phenomenon home (> 8 m/s predictions are 55–65 % of truth); sd 0.02–0.04; same-seed cross-machine offset 0.06 → each machine carries its own baseline.
- 17 interventions, 15 blind to the compression (idw, speed_weight, strat, macro, Huber 0.5, rel_loss, strong T 2.5, β-by-speed, log magnitude, increment heads μ 5/20 …); only full yaw supervision (−0.049) + speed-density reweighting cap 5 (−0.036) pass when stacked (−0.072) → candidate **a3v10_s42**: public 0.3797 (v6 0.3566), official 9/18 0.2881 → rejected. The fold gain sat on the side-flying pair 0023/0024.
- Mechanism correction (`next_steps` evening): out-of-distribution predictions really are low and it is 90 % magnitude; binned by *prediction* the model is calibrated, binned by *truth* it regresses to the mean; the tail's true speed is estimated at 11–16 m/s.
- Engineering: cloud rules set (> 2 h goes to the cloud, lane lists can be appended while running, self-stop on completion).

### 9/18 (Thu) four roads; NeuroBEM; S-fast first batch
- Road 1 (GT-tilt gate) fails, road 2 (calibration targets) merged into road 3, road 3b (linear drag term) raises the fold magnitude but the clean val collapses (0.194 vs 0.179).
- **Road 4, external NeuroBEM** (78 segments, 54 min, 14 cross-correlating segments removed): fold −0.17; train-only official −0.006 (tv_extw03 0.2850); **train+val a3v12 official 0.2819 loses to v6**; the human ATE 0.878 side effect is a body-frame z bias growing with speed → `--ext_until` (last 20–40 epochs competition-only) removes it → a3v13 public 0.35822 (the official query was cancelled after the 22:50 ruling). Lesson: train-only gains ≠ train+val gains (external data only substituted for the val fast flights).
- 22:50 rules: the organisers' written ruling in discussion #737628 ("only train/val may enter the final") → **NeuroBEM out**.
- **Breakthrough analysis**: counterfactual — scale the horizontal specific force of #29/#56 by 0.5/1.5, v6/a3v13 move < 0.3 m/s → **the models do not read drag**; the tail is low-thrust, low-rotation straight flight (training's fast flights are high-thrust, high-rotation manoeuvres); T scales drag by k² and speed by k, teaching the wrong slope. → **S-fast**: drag-consistent translation scaling s ∈ [1, 1.8] on the racing family only (f′ = s(f − g·up) + g·up, v′ = s·v, attitude and gyro unchanged), racing-family T capped at 1.2, `--aug_mix`.
- S-fast first batch (fold A): ruler 2 (drag sensitivity) fails, sprint 3.49 → 3.05 misses 2.7 → "closed" by the plan; **but** at 23:15 the reviewer noted that fold A holds out all six sprints and makes the channel learn the wrong sign → sprint fold RS2: base 2.73 / rest2 2.81 / rest2S 2.49 / **sfast 2.10**. The physics-override probe (`sprint_override.py`, nearest-session coefficient hand-dispatched) scores public 0.3277 — an external module, evidence only, never delivered.

### 9/19 (Fri) a3v16S (S-fast); learned inertial pipeline A→C; a3v19
- 03:35 **`a3v16S_s42` = v6 + S-fast: public 0.3321, official 0.26577** (drone 1.244/0.703; tail 15.1 → 12.7; #30 0.79 → 1.79; car 0.313 / human 0.639 / dog 0.357 inside the band). Rest-reference channels (slowrest/slowrest2) are negative on top of S (a3v15S public 0.3408).
- **Learned inertial pipeline**: A `--grav slowdr` 25 channels (rest_frame_gyro, v_DR, t_since) + `--fuse` gate head; B AttNet (`unified/attnet.py`: block-wise small-angle sums + log-depth prefix products, differentiable SO(3) integration, GRU corrections); C CalNet (`unified/calnet.py`: 64-d recording summary → S/b/τ/gyro-frame map, flip only when "logit > 2 and the rule agrees"). Each sub-module meets its gate (CalNet oracle on the poorly calibrated family 2.4–5.2 → 0.17–0.72; AttNet 0039 4.4° → 1.0°), **no main-network card beats S alone** (sprint fold sp_cal 2.37/2.01 vs 2.10); the fusion gate closes before epoch 30 (trusted windows 5 %). Self-driving cloud pipeline `cloud/ins_pipeline.sh` + `queue_relay.sh`.
- **a3v19_s42 (everything stacked, train+val) official 0.26389 / public 0.3336**: best drone 1.179/0.683 but **car ATE 0.421 out of band** (unbounded v_DR, normalisation std ≈ 3000 leaking).
- Rules: official budget 5/day (UTC); Kaggle 5/day; deadline 9/21 07:55 Taipei. Decision: deliver a3v16S, Kaggle finals a3v16S + a3v19; run the two-path plan on 9/20.

### 9/20 (Sat) two paths → the four roads brought forward → a3v20 → the recursion night
- **Two-path overnight batch**: path 1 done (tv_sfast twin val 0.1784 passes; S cap 1.4 / T probability 0.35 tie on the fold but lower the tail → no a3v16Sb; **a3v16S_s43 official 0.27336**); path 2 six cards produced nothing (logs not synced, /workspace wiped). a3v16S delivery dry-run passes (1:28, 1.39 GB, p99 1.5e-4).
- From 15:30 the four roads were run early (two 5090s + two 3090s): **road 1, three stages** (stage 1 supervised gate: learns "when" but trusted windows 5 %, tv_g val 0.201 collapses; three review rounds fix six bugs — **v_DR always q·tanh bounded + fixed /5 normalisation, 6-D rotation columns, content-md5 cache key, yaw-conjugated EKF increments, inference_mode placement, cat dimension** — all pre-fix slowdr results discarded; stage 2 **EKF recursion head** `--ekf --ekf_sup 0.5` (v_k = ΔR_k v_{k−1} + ΔV_k + K_k(v_direct − v_prop)): sprint fold sp_ekf_s43 1.88 / s42 2.29 (seed sensitive), fold A fa_ekf 1.228 (S 1.377), tv_ekf 0.1821 (S 0.1784); stage 3 CalNet 6-D frame head worse than stage 2); **road 2** (hover spectra) fails the CPU identifiability gate; **road 3** (`--drag_pose`) 2.30–2.38, no gain; **road 4** (`--traj_film`, 35-d descriptors) zero on the folds but **tv_film_s42 val 0.1746, the best clean val ever**.
- 20:11 **`a3v20_s42` = v6 + S-fast + corrected inertial channels (CalNet/AttNet, bounded v_DR, EKF head; train+val): public 0.32793, official 0.25878** (car 0.341, human 0.582, dog 0.345, drone 1.200/0.675; tail 11.84; #30 1.26) — the car leak is gone. The gain is broad, not luck; deliver a3v20 (rule A).
- 20:40 `a3v21_s42` (S-fast + FiLM v1) official 0.26306 / public 0.32695; `a3v20_s43` 0.26933 (car 0.365, dog 0.392 out of band; a3v20 family seed range 0.259–0.269). a3v22 (a3v20 + FiLM) crashed once on a bootstrap missing `moe.py` and was cancelled.
- Delivery package `hf_release_a3v20/` isolated dry-run (80 s, 1.9 GB, max diff 2.0e-3, sub-module path exercised).
- **From 21:40, the recursion study** (22 cards, no official query): §2 E1 (`--ekf_dv int` window-integrated increments + `--ekf_horizon 10` mask + `--ekf_feats` + `--ekf_soft 0.5`) three-fold sprint 1.89/1.62/2.40 = 1.97 (S 2.12, a3v20 recipe 2.07), fold A 1.260, **tv_e1 val 0.1913, every platform worse in ATE and AVE**; ablations carry (dv=0, dR=I) 2.24 and rot 2.25 → the sprint gain needs the integrated increments; carryS (plain 14-channel recursion) tv 0.1828, folds ≈ S; target variants nosup 0.1798 (gain probe ≈ 1.0 = recursion off), margin 0.1887, mask_direct 0.1878 (sprint fold only 2.09), nsmd 0.1815; §3 descriptor v2 (yaw-invariant) tv 0.1797 / fold 2.23, worse; §5 Kabsch has no information on sprints. **Structural conclusion: the EKF gain is one knob with one trade-off (≈1 = S alone; 3 % = a3v20; 12–40 % = lag), and the gain has no input that separates "a sprint near a rest anchor" from "a car starting from rest".**
- CPU follow-ups: anchors (even-IMU racing recordings fine, odd-IMU ones 7–9° from accelerometer bias; the four test tail sequences all anchor at window 0, #57 has no rest); the second drone family's gyro is rate-consistent with −I rather than R_MAP, but a ~2.5 m/s² accelerometer bias dominates and the GT is not a rigid rotation of the IMU frame → parked.

### 9/21 (Sun) wrap-up
- 00:16 `a3v22_s42` trained (moe.py fixed); 01:45 sent to the official service and Kaggle at the user's request: **0.26359 / 0.34375** (car 0.355 out of band, tail 13.07, drone ATE 1.151 better but AVE 0.689 worse) → does not replace a3v20. tv_film_s43 val 0.1770 (≤ 0.178) → Kaggle finals **a3v20 + a3v21**. All four instances stopped 01:30.
- HF: `LexHo/tartanimu-a3v20` (public, commit 2426e014; after re-download, isolated dry-run 85 s / 2.0 GB / max diff 1.9e-3), `LexHo/tartanimu-a3v21` (1b0d0829; 69 s / 1.4 GB / 1.6e-3). Forms submitted; technical report v5.1 (eight investigations) attached.

### 9/28 (Sun) final standings
- The private leaderboard was published: **20th of 131 teams, 0.19912** (`a3v20_s42`, the delivered model). Public leaderboard 18th, 0.32695.
- The private half ranks fourteen of my submissions between 0.1932 and 0.2094 — seven ranks. The best of them on that half would have been 18th (`a3dnTV3td`, the day-7 model); on the full 89-sequence test, which is the better estimator, the delivered model beats both other finalists. The pre-registered rule picked correctly against the estimator it was designed for.
- No certificate is issued: the competition is a Kaggle Community competition (reward "Kudos", no medals or points), and the organisers' only announced recognition is an invitation to the *IMU Foundation Model* white paper for the top 10. Verification links and exports: `docs/results.md` §5 and `docs/evidence/`.

---

## 3. Architecture evolution

### 3.1 The delivered model (a3v20_s42), layer by layer
| Layer | Content | Params |
|---|---|---|
| Pre-processing (per recording, frozen) | **CalNet**: 64-d recording summary (rest statistics, gravity-norm consistency, low-passed tilt-consistency columns 33/34) → accelerometer S(3), b(3), τ, gyro-frame map class (flip when logit > 2 and the rule agrees, R_MAP = (−y,−x,−z)); **AttNet**: rest anchor (the 1-s window minimising |mean‖a‖−9.81| + mean‖ω‖ + 0.5·std‖a‖), SO(3) gyro integration (block-wise small-angle sums + log-depth prefix products) + GRU corrections → R_{t→F0} | 26 k + 52 k |
| Derived channels (`derived_channels`, 13 → 17 columns) | rest-frame up (3), **v_DR = q·tanh(∫(R f_cal + g)/q), q = 8** (3), t_since/10 (1), first two columns of R in 6-D (6); added on 9/20 night: v_F0 (3) and calibration deviation (1) | — |
| 25 input channels | raw 6 + slow complementary-filter up decomposition 8 + rest reference 4 (f_rest 3, score) + rest-up gyro 3 + v_DR 3 + t_since 1; fixed normalisation v_DR std 5, t_since 1 | — |
| trunk | 1-D ResNet, 7 dilated residual blocks, 192 ch × 25 steps, BatchNorm; `--init_trunk dense180_tv_s42.pt --init_trunk_pad` (14 → 25 channels, zero-init) | ~1.5 M |
| tokens | adaptive pool per window → **5 tokens** + phase embedding | |
| context | **bidirectional GRU × 2, K = 40 windows** (40 s), mixer hidden 128 + **wide branch** (5-s low-pass summary) | |
| heads | direct linear head 256 → 3 + **EKF recursion head**: v_k = ΔR_k v_{k−1} + ΔV_k + K_k(v_direct − v_prop), ΔR/ΔV from window-end frames (`ekf_increments`), K_k = σ(Linear(h_k, t_since, score, log1p‖v_DR‖)), bias +3; auxiliary platform logit (0.05, discarded at inference) | |
| total | | **2,166,242** (main) + 77,585 (sub-modules) |

**Training**: dense random-offset windows (never crossing trajectories), platform-uniform, AdamW 2e-3, OneCycle, **120 epochs × 160 steps, batch 52 chunks × 40 windows**, dropout 0.2, vector Huber β 0.05 (fused output + direct head 0.5×), EKF gain BCE 0.5, AMP; **train+val, last epoch**, seed 42 pre-designated. **Augmentation**: T 0.7–1.5 (drone, p 0.5; identity racing family capped at 1.2), **S-fast 1.0–1.8** (racing family, `--aug_mix` half/half with T), yaw-direction supervision 0.5 (identity source). **Inference**: per trajectory, stride-K/2 overlapping chunks, each window from the chunk whose centre is nearest; valid-length for short trajectories; deterministic, no TTA.

### 3.2 Version table (each row's difference to the previous; official = full test)
| Version | Date | Difference | val (train-only twin) | official | public |
|---|---|---|---|---|---|
| official unified.pt | 9/8 | 4-head ResNet-LSTM, forced head | 0.419 (GT routing) | — | 0.937 |
| single-window CNN → g20d | 9/8 | own trunk; dense windowing; 20-window GRU | 0.232 | — | 0.406 |
| f5 | 9/9 | fine=5 (5 tokens/window) | 0.219 | — | 0.391 |
| all3 | 9/12 | + slow gravity 8 ch (14 ch) + Huber 0.05 + K=40 | 0.2096 | 0.3057 | 0.3694 |
| a3sslc / a3dn | 9/13 | + masked-IMU SSL trunk (dense, 2 h) | 0.1962 | 0.2962 (sslc) | 0.3759 |
| v1 a3dnTV | 9/13 | train+val pre-training + fine-tuning | — | 0.2837 | 0.3702 |
| **v2 a3dnTV3td** | 9/14 | + T 0.7–1.5 (3-h SSL) | 0.1806 (td0715) | **0.2687** | 0.3526 |
| v3 …A1 | 9/14 | fixed T (short trajectories kept) | 0.1802 | 0.2743 | 0.3639 |
| a3v4 / a3v5 | 9/15 | + wide (R3); dense120 / dense180_tv | — | 0.2773 / 0.2831 | — |
| **v6 a3v6** | 9/15 | + yaw-direction supervision 0.5; dense180_tv | 0.1770 (R3yawd) | **0.2702** (s43–45 0.279–0.281) | 0.3566 |
| a3v7/7f/8/9/10 | 9/15–17 | T variants / paired yaw / full yaw + density reweighting | | 0.2842/0.2784/0.2750/0.2719/0.2881 | a3v10 0.3797 |
| d16/d17 series | 9/16–17 | train-only: SSL 3.3×, JEPA, synthetic | 0.1753–0.1845 | 0.2831 (base) / 0.2868 / 0.2973 / 0.3010 | |
| a3v12 / a3v13 | 9/18 | + NeuroBEM external (ext 0.3 / + ext_until) | — | 0.2819 / (not queried) | 0.3582 |
| a3v15S | 9/18 | + rest-reference channels + S-fast | | — | 0.3408 |
| **a3v16S** | 9/19 | v6 + **S-fast** (T_fast 1.2, S 1.0–1.8, aug_mix) | 0.1784 (tv_sfast) | **0.26577** (s43 0.27336) | 0.3321 |
| a3v19 | 9/19 | + inertial channels (unbounded v_DR, fuse gate) | 0.1833 (tv_cal) | 0.26389 (car 0.421) | 0.3336 |
| **a3v20** | 9/20 | + corrected inertial channels (bounded v_DR, 6-D, EKF head) | 0.1821 (tv_ekf) | **0.25878** (s43 0.26933) | 0.32793 |
| a3v21 | 9/20 | a3v16S + trajectory FiLM v1 | 0.1746 / 0.1770 (s42/s43) | 0.26306 | 0.32695 |
| a3v22 | 9/21 | a3v20 + trajectory FiLM v1 | — | 0.26359 | 0.34375 |

### 3.3 Per-platform official trajectory (ATE20 / AVE)
| Version | car | human | dog | drone | tail #29/#56/#37/#38 (sum) |
|---|---|---|---|---|---|
| all3 (9/12) | 0.340/0.066 | 0.629/0.058 | 0.383/0.078 | 1.524/0.844 | 4.48/4.24/2.35/1.78 (12.9) |
| a3sslc (9/13) | 0.331/0.069 | 0.637/0.057 | 0.396/0.080 | 1.465/0.801 | 3.99/4.76/2.95/2.24 (13.9) |
| v2 a3dnTV3td (9/14) | 0.330/0.067 | 0.654/0.056 | 0.347/0.076 | 1.258/0.711 | 4.64/4.46/2.99/1.88 (14.0) |
| v6 a3v6 (9/15) | 0.337/0.067 | 0.603/0.053 | 0.357/0.077 | 1.274/0.723 | 5.25/4.96/2.88/2.00 (15.1) |
| a3v16S (9/19) | 0.313/0.067 | 0.639/0.055 | 0.357/0.076 | 1.244/0.703 | 4.42/4.28/2.45/1.55 (12.7) |
| a3v19 (9/19) | 0.421/0.075 | 0.624/0.055 | 0.333/0.079 | 1.179/0.683 | 3.86/4.07/2.42/1.85 (12.2) |
| **a3v20 (9/20)** | 0.341/0.070 | 0.582/0.055 | 0.345/0.081 | 1.200/0.675 | 4.14/3.84/2.38/1.49 (11.8) |
| a3v21 (9/20) | 0.346/0.072 | 0.632/0.054 | 0.343/0.076 | 1.225/0.687 | 4.35/4.05/2.20/1.60 (12.2) |
| a3v22 (9/21) | 0.355/0.075 | 0.606/0.055 | 0.375/0.083 | 1.151/0.689 | 4.39/4.25/2.39/2.03 (13.1) |
| competitor AxisTilted2 | 0.363/0.070 | 0.545/0.045 | 0.330/0.074 | 1.002/0.515 | 3.23/3.29/—/— |

Reading: all progress from 9/12 to 9/20 is drone (0.844 → 0.675 AVE); T lowers the body by −16 % but raises the tail by 0.5–1.0 (v2 → v6 tail 14.0 → 15.1); S-fast pulls the tail back to 12.7; the inertial channels take it to 11.8 and fix human/dog; car stays in 0.31–0.34, human 0.58–0.65, dog 0.33–0.39 with no structural change.

---

## 4. Evolution of the evaluation protocol (why the ruler changed)
1. **9/8–9/9**: paired bootstrap + multi-seed; val selects models (same architecture 1:1, cross-architecture 3×); val is optimistic by 0.18 in absolute terms.
2. **9/12–9/13**: the official scoring service gives per-platform + per-sequence on the full test; public and full test once disagreed (a3sslc); the val→test gap is entirely drone and the test drone flies harder. Decision: select on clean val day to day, use the official service only to confirm pre-frozen candidates.
3. **9/14**: the official per-sequence table exposes the tail #29/#56 (no such val trajectory) → val is blind to the tail.
4. **9/15**: hold-out (12 recordings) — rerun variance 0.14, too noisy.
5. **9/17**: **fold A** (26 recordings, both IMUs of a flight held out together, no > 8 m/s in training) sd 0.02–0.04; each machine carries its own baseline; report by bins (> 8 m/s compression ratio, high-rotation, the rest).
6. **9/18 night**: holding out all six sprints makes a channel learn the wrong sign → **sprint fold** (one sprint pair + one aggressive pair held out; the other sprints stay in training); extended on 9/20 night to three rotating folds.
7. **9/20 night (reviewer)**: the three rulers measure three populations (sprint fold = A/B, fold A = C + extrapolation, val = D + the other platforms) and the official score is their weighted mix → **use each ruler as a gate (threshold = 2× seed sd), never as a ranker**; candidates always two seeds; a seed range > 0.008 counts as unstable.
8. **Selection rule A**: the delivered seed is pre-designated (42), the second seed only reports the range; recipe-level choices use the official full test (disclosed as feedback use, 33 queries in total).

---

## 5. Ledgers (generated by script from `runs/official/*.csv` and the Kaggle API, 2026-09-21 12:00)

### 5.1 Official full-test queries (33, sorted by score)
| tag | official | car ATE/AVE | human | dog | drone | tail #29/#56/#37/#38 AVE | committed |
|---|---|---|---|---|---|---|---|
| a3v20_s42 | 0.25878 | 0.341/0.070 | 0.582/0.055 | 0.345/0.081 | 1.200/0.675 | 4.14 / 3.84 / 2.38 / 1.49 \| #30 1.26 | 09/20 20:13 |
| a3v21_s42 | 0.26306 | 0.346/0.072 | 0.632/0.054 | 0.343/0.076 | 1.225/0.687 | 4.35 / 4.05 / 2.20 / 1.60 \| #30 1.16 | 09/20 20:40 |
| a3v22_s42 | 0.26359 | 0.355/0.075 | 0.606/0.055 | 0.375/0.083 | 1.151/0.689 | 4.39 / 4.25 / 2.39 / 2.03 \| #30 1.13 | 09/21 01:40 |
| a3v19_s42 | 0.26389 | 0.421/0.075 | 0.624/0.055 | 0.333/0.079 | 1.179/0.683 | 3.86 / 4.07 / 2.42 / 1.85 \| #30 1.00 | 09/19 08:03 |
| a3v16S_s42 | 0.26577 | 0.313/0.067 | 0.639/0.055 | 0.357/0.076 | 1.244/0.703 | 4.42 / 4.28 / 2.45 / 1.55 \| #30 1.79 | 09/19 08:03 |
| a3dnTV3td_s42 | 0.26871 | 0.330/0.067 | 0.654/0.056 | 0.347/0.076 | 1.258/0.711 | 4.64 / 4.46 / 2.99 / 1.88 \| #30 0.78 | 09/17 12:11 |
| a3v20_s43 | 0.26933 | 0.365/0.071 | 0.592/0.054 | 0.392/0.083 | 1.253/0.703 | 3.74 / 4.38 / 2.47 / 2.59 \| #30 1.17 | 09/21 01:40 |
| a3v6_s42 | 0.27017 | 0.337/0.067 | 0.603/0.053 | 0.357/0.077 | 1.274/0.723 | 5.25 / 4.96 / 2.88 / 2.00 \| #30 0.79 | 09/17 12:11 |
| a3v9_s42 | 0.27190 | 0.332/0.066 | 0.670/0.055 | 0.378/0.077 | 1.254/0.721 | 5.12 / 5.08 / 3.06 / 1.52 \| #30 0.58 | 09/17 12:11 |
| a3v16S_s43 | 0.27336 | 0.328/0.066 | 0.620/0.055 | 0.349/0.076 | 1.279/0.738 | 4.35 / 4.49 / 2.81 / 2.37 \| #30 1.56 | 09/20 14:20 |
| a3dnTV3tdA1_s42 | 0.27426 | 0.291/0.064 | 0.590/0.053 | 0.370/0.076 | 1.297/0.751 | 5.18 / 5.20 / 2.77 / 2.15 \| #30 0.59 | 09/17 12:11 |
| a3v8_s42 | 0.27504 | 0.330/0.066 | 0.660/0.056 | 0.365/0.077 | 1.284/0.735 | 5.27 / 4.86 / 2.74 / 1.63 \| #30 1.14 | 09/17 12:11 |
| a3v4_s42 | 0.27726 | 0.331/0.067 | 0.688/0.058 | 0.343/0.076 | 1.292/0.741 | 5.39 / 4.76 / 2.64 / 2.68 \| #30 0.77 | 09/17 12:11 |
| a3v7f_s42 | 0.27835 | 0.313/0.067 | 0.618/0.054 | 0.367/0.077 | 1.317/0.755 | 5.38 / 5.49 / 2.46 / 1.73 \| #30 0.90 | 09/17 12:11 |
| a3v6_s43 | 0.27919 | 0.333/0.069 | 0.605/0.054 | 0.354/0.077 | 1.310/0.760 | 4.64 / 5.08 / 3.37 / 2.92 \| #30 1.05 | 09/17 12:11 |
| a3v6_s44 | 0.27932 | 0.283/0.064 | 0.639/0.055 | 0.352/0.075 | 1.335/0.766 | 4.86 / 4.58 / 3.37 / 2.86 \| #30 1.56 | 09/17 12:11 |
| a3v6_s45 | 0.28092 | 0.333/0.067 | 0.645/0.055 | 0.350/0.075 | 1.343/0.760 | 5.25 / 5.00 / 2.53 / 1.92 \| #30 1.90 | 09/17 12:11 |
| a3v12_s42 | 0.28187 | 0.299/0.063 | 0.878/0.064 | 0.350/0.077 | 1.318/0.731 | 5.08 / 5.05 / 2.90 / 2.53 \| #30 1.03 | 09/18 14:02 |
| d16_full_base_s42 | 0.28311 | 0.370/0.070 | 0.608/0.054 | 0.381/0.077 | 1.332/0.764 | 5.02 / 4.95 / 3.15 / 2.89 \| #30 0.57 | 09/17 12:11 |
| a3v5_s42 | 0.28313 | 0.326/0.067 | 0.638/0.055 | 0.372/0.076 | 1.344/0.769 | 5.22 / 5.05 / 3.03 / 2.71 \| #30 1.08 | 09/17 12:11 |
| a3dnTV_s42 | 0.28374 | 0.341/0.069 | 0.652/0.056 | 0.354/0.076 | 1.361/0.764 | 4.10 / 4.31 / 2.74 / 2.47 \| #30 1.01 | 09/17 12:11 |
| a3v7_s42 | 0.28419 | 0.394/0.072 | 0.669/0.056 | 0.347/0.078 | 1.335/0.757 | 5.26 / 5.46 / 2.98 / 1.61 \| #30 1.02 | 09/17 12:11 |
| tv_extw03_s42 | 0.28495 | 0.425/0.077 | 0.639/0.056 | 0.388/0.079 | 1.350/0.745 | 4.86 / 5.16 / 2.72 / 2.12 \| #30 1.48 | 09/18 13:32 |
| d17_ssl600_s42 | 0.28677 | 0.326/0.068 | 0.626/0.055 | 0.394/0.077 | 1.390/0.775 | 5.28 / 5.25 / 3.02 / 2.70 \| #30 0.71 | 09/17 12:11 |
| a3v10_s42 | 0.28812 | 0.370/0.072 | 0.679/0.060 | 0.369/0.078 | 1.338/0.768 | 5.22 / 5.17 / 2.52 / 2.18 \| #30 0.95 | 09/20 20:40 |
| d17_ssl600tm03_s42 | 0.28950 | 0.382/0.075 | 0.660/0.056 | 0.418/0.079 | 1.375/0.764 | 5.16 / 4.85 / 2.95 / 2.27 \| #30 1.14 | 09/17 12:11 |
| tv_base_s42 | 0.29065 | 0.338/0.068 | 0.618/0.055 | 0.384/0.077 | 1.394/0.794 | 5.84 / 5.33 / 3.34 / 1.79 \| #30 1.90 | 09/20 20:40 |
| a3sslc_s42 | 0.29618 | 0.331/0.069 | 0.637/0.057 | 0.396/0.080 | 1.465/0.801 | 3.99 / 4.76 / 2.95 / 2.24 \| #30 1.24 | 09/17 12:11 |
| d17_jepa_s42 | 0.29731 | 0.355/0.067 | 0.589/0.054 | 0.358/0.076 | 1.439/0.829 | 4.79 / 5.03 / 3.19 / 2.70 \| #30 1.67 | 09/17 12:11 |
| tv_ext_s42 | 0.29744 | 0.391/0.072 | 0.821/0.065 | 0.358/0.077 | 1.364/0.784 | 5.07 / 4.98 / 3.15 / 2.81 \| #30 1.62 | 09/20 20:40 |
| a3ssl_s42 | 0.29838 | 0.325/0.068 | 0.662/0.058 | 0.402/0.081 | 1.464/0.807 | 4.41 / 4.11 / 2.94 / 2.09 \| #30 1.94 | 09/17 12:11 |
| d17_synth4_s42 | 0.30101 | 0.330/0.067 | 0.646/0.055 | 0.375/0.077 | 1.466/0.833 | 5.08 / 5.19 / 3.59 / 3.50 \| #30 1.86 | 09/17 12:11 |
| all3_s42 | 0.30571 | 0.340/0.066 | 0.629/0.058 | 0.383/0.078 | 1.524/0.844 | 4.48 / 4.24 / 2.35 / 1.78 \| #30 2.22 | 09/17 12:11 |

(a3v13_s42/s43 were scheduled for 9/19 08:04 and cancelled after the 9/18 22:50 ruling; the 33 rows above are every official query.)

### 5.2 Kaggle submissions (26)
| # | date (Taipei) | file | public | note |
|---|---|---|---|---|
| 1 | 09/08 15:45 | submission_zero.csv | 1.05383 | |
| 2–5 | 09/08 16:27 | sub_test_{drone,car,dog,human}.csv | 1.865 / 0.937 / 0.969 / 0.967 | official baseline unified.pt, forced head |
| 6 | 09/09 12:41 | sub_test_g20d_s45.csv | 0.40598 | dense offsets + 20-window GRU, seed 45, val 0.2277 |
| 7 | 09/09 12:41 | sub_test_g20d_s43.csv | 0.41410 | same, seed 43, val 0.2356 |
| 8 | 09/09 22:36 | sub_test_f5_s42.csv | 0.39096 | fine mixer 5 sub-steps, val 0.2176 |
| 9 | 09/09 22:36 | sub_test_dec_s42.csv | 0.40612 | fine mixer + segment/deviation decomposition |
| 10 | 09/10 14:52 | sub_test_mf4_s42.csv | 0.39744 | FiLM trunk conditioning on segment statistics |
| 11 | 09/12 18:07 | sub_test_all3_s42.csv | 0.36940 | slow gravity + Huber 0.05 + 40 windows, val 0.2093 |
| 12 | 09/13 09:15 | sub_test_a3sslc_s42.csv | 0.37593 | masked-IMU pre-trained trunk (train+val) |
| 13 | 09/13 20:23 | sub_test_a3dnTV_s42.csv | 0.37018 | dense pre-training (train+val, 90 min) + all3 |
| 14 | 09/14 14:18 | sub_test_a3dnTV3td_s42.csv | 0.35260 | 3-h pre-training + train+val + time dilation |
| 15 | 09/14 16:14 | sub_test_a3dnTV3tdA1_s42.csv | 0.36392 | v2 + time dilation reaching short trajectories |
| 16 | 09/15 15:44 | sub_test_a3v6_s42.csv | 0.35660 | v6 |
| 17 | 09/15 21:36 | sub_test_C0_s42.csv | 0.37818 | train-only warm-start candidate |
| 18 | 09/17 22:04 | sub_test_a3v10_s42.csv | 0.37968 | v6 + full yaw supervision + speed-density reweighting |
| 19 | 09/18 17:26 | sub_test_a3v13_s42.csv | 0.35822 | v6 + NeuroBEM external (later ruled out) |
| 20 | 09/18 19:18 | sub_a3v6_override_0037.csv | 0.32770 | physics sprint override probe (not a submission candidate) |
| 21 | 09/18 22:21 | sub_test_a3v15S_s42.csv | 0.34077 | v6 + rest-reference channels + S-fast |
| 22 | 09/18 23:54 | sub_test_a3v16S_s42.csv | 0.33210 | v6 + S-fast |
| 23 | 09/19 07:13 | sub_test_a3v19_s42.csv | 0.33362 | + learned-INS channels (first version) |
| 24 | 09/20 20:11 | sub_test_a3v20_s42.csv | 0.32793 | + corrected inertial channels + EKF head |
| 25 | 09/20 20:34 | sub_test_a3v21_s42.csv | 0.32695 | S-fast + trajectory FiLM |
| 26 | 09/21 01:39 | sub_test_a3v22_s42.csv | 0.34375 | a3v20 + trajectory FiLM |

### 5.3 GPU usage (estimate)
| Resource | Period | Usage |
|---|---|---|
| local GPU | 9/8–9/17 | 8–20 min per run, ≈ 300 runs, ≈ 60 h |
| Kaggle T4 (free) | 9/9–9/16 | MoE, equivariance, gsP seeds, YA/YB, AA, ≈ 30 kernel-h |
| cloud RTX 3090 ×2 | 9/16 night–9/21 | ≈ 45 h |
| cloud RTX 4090 ×2 | 9/16 night | node failure, refunded |
| cloud RTX 5090 ×2 | 9/18–9/21 | ≈ 30 h; 20–30 min per card |
| SSL pre-training | 9/13, 9/16 | dense120 2 h, dense180_tv 3 h, dense600 5 h |

---

## 6. Catalogue of negative results (each with the number that killed it; all beyond the seed threshold or with a mechanism)

### 6.1 Architecture
| Idea | Number | Why |
|---|---|---|
| official 4-head model + any label-free routing | public 0.937 | no mechanism to infer the platform |
| Transformer mixer | +0.049 | large seed variance |
| LSTM trunk | +0.009 | |
| continuous 20-s trunk | +0.031 | BatchNorm polluted by short-trajectory mini-groups 0.021 + receptive field 0.014 |
| O(2)-equivariant canonical frame (EqNIO) | +0.24 (equivariance 1e-15) | premise (accurate gravity, 5.5°) fails; training loss 27× |
| MoE in four placements (head K=4/8, MLP, FiLM) | 0 | |
| local cross-attention AT1/AT2 (warm + full) | 0 | |
| per-window increment-correction head LR0 | 0 | |
| 512-bin expectation head | val −0.001, hold-out +0.014 | |
| increment heads μ 5/20 (fold A) | +0.044/+0.094 | the increment head sees the same out-of-distribution features as the direct head |
| linear drag term (road 3b) | fold magnitude ratio 0.55 → 0.72, val 0.194 vs 0.179 | the linear term carries magnitude in-distribution too |
| drag-consistent attitude synthesis `--drag_pose` (road 3) | sprint fold 2.30–2.38 vs S 2.10 | |
| CalNet 6-D frame head (road 1 stage 3) | 2.16–2.29 vs stage 2 1.88; training frame error 28.9° | a 64-d summary cannot predict a continuous rotation |
| descriptor v2 (yaw-invariant FiLM) | tv 0.1797 vs v1 0.1746; fold 2.23 | the body-frame means are exactly what FiLM uses to recognise a recording |
| capacity: trunk 0.5×/1.5×, mixer 128–384, width 1.5 (three settings) | 0 / +0.006 | |
| 60-window context | +0.015 (training loss K=20/40/60 = 0.0097/0.0047/0.0023) | longer memorises more |
| GroupNorm | +0.008 | |

### 6.2 Losses, targets, regularisation
| Idea | Number |
|---|---|
| differentiable ATE20 surrogate w 0.1–3 | monotonically worse |
| per-frame velocity supervision (three rounds) | 0 |
| gravity-direction auxiliary head | 0 |
| scale-aware output | +0.020 |
| additive segment/deviation decomposition | +0.004 (70 % from ATE20) |
| GradNorm α 1.5 | +0.002 (drone weight pushed to 0.19) |
| Huber β 0.5 / β by speed / relative error / log magnitude (fold A) | +0.004 / +0.045 / +0.014 / +0.099 |
| speed-density reweighting cap 5 + full yaw supervision (fold A −0.072) | official a3v10 0.2881 vs v6 0.2702 | fold gain concentrated on two side-flying recordings, no transfer |
| dropout 0.4/0.5, weight decay, white noise, band randomisation, RSC, SAM, macro loss, no platform CE, 180/240 epochs, lr 1e-3/4e-3 | all null or worse |
| EMA, SWA, cross-run soup (collapses 0.3459), same-basin soup, Lookahead | 0 / collapse / 0 / +0.003 |
| EKF gain-target variants: soft σ(Δe/0.5), margin 0.15, mask→direct, no supervision, no-sup + mask | tv 0.1913 / 0.1887 / 0.1878 / 0.1798 / 0.1815 (S 0.1784) | gain open = lag, closed = S alone |

### 6.3 Augmentation and data
| Idea | Number |
|---|---|
| platform sampling ratios, drone 2×, aggressive 3×, identity 3×, idw 3, speed_weight, strat | 0 or worse (more exposure = memorisation) |
| bias-perturbation (official parameters), acc gain ×U(0.9,1.1) | worse / +0.0024 |
| random body-z yaw (full supervision) | 0.677 (training Huber 0.065) | the body x/y identity is a signal (mounting yaw offset) |
| time-warp TTA, phase-jitter TTA, 8-yaw TTA (not run) | all worse |
| strong T (identity 1.0–2.0 / 1.2–2.5, k to 2.5) | identity 0.663 → 0.737 | pulls toward a direction val does not have |
| translation scaling S 0.8–1.25 (four platforms) | −0.008, does not add to T | |
| S-fast cap 2.2 / 1.4, T probability 0.35 | tie on the fold, lower tail predictions | not a dose problem |
| SSL 3.3× (dense600), JEPA | official 0.2868 / 0.2973 vs 0.2831 | |
| synthetic aggressive flights (synth4 and mild/half/out/fast) | official 0.3010 | |
| external UZH-FPV | = competition data incl. test → banned, deleted |
| external NeuroBEM (ext 0.3, ext_until) | train+val 0.2819 vs 0.2702; human ATE 0.878; ruled out |
| segment-level / joint reconstruction / 60-epoch / 50 %-mask pre-training | 0.2034 / 0.2390 / 0 / 0 |

### 6.4 Physics and inertial channels
| Idea | Number |
|---|---|
| pure body-frame integration (GT gravity + GT initial velocity) | 1 s 0.659, 5 s 2.23, 20 s 6.96 (model 0.219) |
| preint, `--grav adapt`, GT-up diagnostic | 0 / 0 / hold-out 1.227 |
| rest-reference channels slowrest/slowrest2 | fold A fails; a3v15S public 0.3408 > a3v16S 0.3321 |
| v_DR + fuse gate (stage 1) | gate closes before epoch 30; trusted windows 5 %; tv_g 0.201 |
| GT relative-motion oracle (5 s) | hold-out 1.14 → 0.56 (room exists), raw hard reconstruction 7.6 |
| physics override (nearest-session coefficient) | public 0.3277 — external module, never delivered |
| closed-form Kabsch gyro→accelerometer frame | sprint residual ~2 (thrust-dominated, no information) |
| second drone family gyro map −I | rate-level right (0.12 vs 1.0), integrated up dominated by a 2.5 m/s² accelerometer bias |
| carry / rot ablations (dv=0) | sprint fold 2.24 / 2.25 vs E1 1.89 | the sprint gain comes from the integrated increments |
| TTT (AdaBN, masked recon) | all negative; and conflicts with frozen weights |

---

## 7. Compliance / disclosure ledger
| Event | Date | Handling | Wording in the report |
|---|---|---|---|
| SSL branch with unlabelled test IMU (a3ssl) | 9/12–13 | queried once 0.2984; excluded on 9/14 after the organisers' reply; no derived weights | "one discarded early branch pre-trained on train+val+test raw IMU, scored once, excluded" |
| organisers' checkpoint as initialisation | judged usable 9/13; reply on 9/14: not allowed | never used in any delivery | — |
| UZH-FPV | 9/16 | leakage check = competition data incl. test → deleted unused | "found to coincide with competition recordings incl. test; deleted unused" |
| NeuroBEM | 9/18 | supervised fine-tuning experiments after removing cross-correlated segments; official queries a3v12/tv_ext*/a3v13; discussion #737628 ruling → not a final candidate | "disclosed; no such model is a final candidate" |
| physics override probe | 9/18 | public once 0.3277; not part of any submission | "a physics post-processing probe, not part of any submission" |
| feedback usage | throughout | official full test 33 queries, Kaggle 26 submissions; recipe-level selection on the official service, seed pre-designated | compliance answer 4 |
| GT attitude during training | from 9/14 | only in augmentation (up for T/S) and sub-module targets; inference from raw IMU | answers 4/5 |
| TTA / ensemble / TTT | — | all tested, none delivered | answer 3 |

---

## 8. Artefact index
| Category | Path |
|---|---|
| delivery packages | HF `LexHo/tartanimu-a3v20`, `LexHo/tartanimu-a3v21` (predict.py, requirements, README, SHA256SUMS, weights/, submission.csv, unified/) |
| official results | `runs/official/<tag>_{overall,platform,sequences,sequences_file}.csv` |
| Kaggle CSVs | the HF repos' `submission.csv` (a3v20 md5 545d7cdd…, a3v21 a53e196f…) |
| core code | `unified/train_v2.py` (all flags), `dense.py` (sampling and augmentation), `updir.py` (up channels, derived channels, caches), `fine_context.py` (trunk/GRU/heads/EKF), `attnet.py`, `calnet.py`, `moe.py` (descriptors/FiLM), `predict.py` |
| rulers | `local_eval/score_val.py`, `sprint_check.py`, `stress_bins.py`, `drag_sensitivity.py`, `official_score.py`; folds `runs/folds/*.json` |
| cloud | `cloud/bootstrap_full.sh`, `launch_lane.sh`, `ss_lane.sh`, `collect_lanes.sh`, `gputw_exec.sh` |
| documents | `docs/workflow_playbook.md`, `docs/recursion_gain_study.md`, `docs/peer_survey.md`, `docs/advice_after_v20.md`, `docs/canonical_numbers.md`; `report/` (LaTeX + PDF) |

---

## 9. What we would do differently
1. **Rulers before methods**: val was blind to groups A/B/C; the folds only existed from 9/17; part of the thirty-plus nulls of 9/8–9/16 were nulls of the ruler, not of the idea.
2. **Hit where the gap is**: the last four days went entirely into the tail (A+B, a fifth of the gap); four fifths of the gap is the 41 ordinary drone flights (D+C) — the best public entry (AxisTilted2) gets drone AVE to 0.515 (ours 0.675) with 64-s contexts, a bias/gravity-state smoother, spectral features and a segment-level ATE loss.
3. **Physics in a smoother, not in a forward recursion**: our EKF gain lags as soon as it opens; the top entry solves one least-squares problem with bias states over the whole context.
4. **Time warping up to 2.0 and short 32-window chunks** (BShankar) — cheap and verifiable.
5. **Honest disclosure of seed selection**: both top competitors picked the best of six seeds; our rule A pre-designated the seed, and a3v20 s42 happened to be the good one (family 0.259–0.269).
6. Process: self-stop on completion, not on a timer; logs into persistent storage; bootstrap installs every file; `pkill -f` kills its own shell; Seq ID = file index + 1.
