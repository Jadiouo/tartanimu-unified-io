# After v20: resolving the contradiction between rulers, and where to go next (evening of 2026-09-20)

Inputs: the four-roads results, `runs/official/a3v20_s42 / s43, a3v21_s42`, the fold and twin CSVs, the road-1 code review §6. Only evidence and ordering here; the schedule is the owner's decision.

## 0. Conclusions
1. The three rulers do not contradict each other: they measure **three different populations**, and the official score is a fixed-weight mixture of those populations. Treating them as "ranking rulers" makes them fight; treating them as "gates per population" makes them consistent.
2. v20 has exactly one real problem: **the EKF head doubles the seed variance** (0.259–0.269), and the cause is in the diagnostics (propagation beats the direct head on only 3 % of windows, p99 error 15 m/s). Fixing three things in the EKF (dv source, innovation feature, soft target) is the next main line.
3. Order: fix the EKF variance → stack FiLM (fix the descriptor first) → car mask → frame estimation for the badly calibrated IMU (the last segment of the tail). Each step is judged with "population gates + two seeds"; only gated candidates go to the official service.

---

## 1. Why the three rulers "contradict"

| Ruler | Population | Test group it maps to | Noise of one measurement |
|---|---|---|---:|
| sprint fold (0037/0039) | 2 sequences, ~120 windows, one session held out | groups A/B (4 sequences, 44 % of drone AVE) | seed sd ≈ 0.3 (1.88 vs 2.29) |
| fold A (26 sequences, all sprints held out) | high rotation + pure extrapolation | group C (11 sequences) and extrapolation ability | sd ≈ 0.03 |
| clean val (train-only twins) | 80 sequences, no sprints, no high rotation | group D (30 sequences) + car/dog/human | sd ≈ 0.002 |
| official full test | 89 sequences | everything, drone 69 % | same-recipe seed sd ≈ 0.005 |

Decomposing a3v20 vs a3v16S with the official formula: drone −0.0072 (of which the four tail sequences 12.70 → 11.84 contribute about −0.005, group D 0.379 → 0.367 about −0.002, ATE improves on 26/45), human −0.0019, car +0.0016, dog +0.0006. Each ruler is right in its own terms:
- Clean val says the EKF makes group D worse by 0.015 (tv_ekf drone 0.374 vs 0.359) — on the official test that is a +0.01-class change on 30 sequences, covered by the tail's −0.86; and v6's four-seed band on group D is 0.338–0.396, so 0.012 is inside the band. **val is right about group D; group D just does not decide the score.**
- Fold A ranks fa_dr2 > fa_ekf: fold A's training set contains no sprint, so the EKF gain cannot learn "when to trust propagation"; it measures the channel's own help on high rotation (0.73 → 0.6, of which official group C 0.688 → 0.681 realises only a small part). **Fold A measures group C and extrapolation, not the tail.**
- The sprint fold splits by seed: two sequences cannot resolve differences below 0.2; it can only be a "is there a large signal" gate, never a ranker.

### 1.1 How to use them from now on
- **No ruler ranks anything any more**; each ruler is a gate with threshold = 2× its seed sd: sprint fold −0.6 (mean of two seeds), fold A −0.06, clean val +0.003 with car/dog/human each ≤ the top of the band. A candidate must not regress on any gate and must pass at least one before it goes to the official service.
- **Extend the sprint fold to three folds**: the three outdoor forward-flight pairs (0037/39, 0038/40, 0041/42) held out in turn, one seed per fold = three independent tail estimates at the cost of three seeds but covering three flights; report the three-fold mean, sd drops to ~0.15.
- **Candidates always two seeds** (s42 pre-designated for delivery, s43 reports the range); a recipe with a seed range > 0.008 counts as unstable and is stabilised before it is sent.
- Official queries already exceed 25, so selection bias exists; the compliance answers say so, and selection rule A stays.

---

## 2. Main line: pushing the EKF variance down

Diagnostics (a3v20, last three epochs): `pbetter` 3.1 %, `kprop` 0.11, `vprop_p99` 15.3 m/s. Meaning: the propagated v_prop beats the direct head on only 3 % of windows and is wrong by 15 m/s on 1 %; the gain head has to learn "almost always closed" on an extremely noisy target, and different seeds close it differently → 0.259 vs 0.269. Three fixes (all in `train_v2.ekf_increments` and the EKF section of `fine_context`, each under half a day):

1. **Change the dv source**: currently dv_k = v_DR(end of window k) − dR·v_DR(end of window k−1), which eats the attitude error of two single frames. Change to a window integral: dv_k = R_kᵀ·∫_{window k} (R f_cal + g) dt (one extra derived column, computed in numpy, always small, no drift accumulation).
2. **Confidence mask**: CalNet's |S−1|, |b|, map logit and seconds since the anchor decide whether dv is usable; unusable windows get dv=0, dR=I (propagation degrades to "carry the previous window"), and the mask becomes the gain's fourth feature.
3. **Innovation and soft target**: add the window's |v_direct − v_prop| to the gain features; change the supervision target from 0/1 to σ((e_direct − e_prop)/0.5).

Acceptance (gates of §1.1): `vprop_p99` < 5, `pbetter` > 10 %, `kprop` on pbetter windows > 0.5; sprint three-fold mean ≤ 1.9; two-seed official range ≤ 0.006. Expectation: not necessarily a lower total, but turning the 0.259 "good seed" into the family median.

## 3. Stacking FiLM (road 4)
a3v21 official 0.2631 (−0.003 vs a3v16S), as val predicted; it improves group D, a different group from v20's gains (tail, drone ATE, human), so in principle additive. Fix the descriptor first: the 6 body-frame vectors among the 40 dimensions (f_rest, calm_mean) do not rotate with yaw augmentation — either multiply by Rz in the yaw branch or replace with |f_rest| and tilt scalars. Then a3v22 with two seeds. Expected −0.002 to −0.004.

## 4. The car band edge
0.341 comes from #81 (+0.18) and #76 (+0.13): residual v_DR on long recordings (bounded but not masked). `--dr_horizon 10` (channel level) with the confidence mask of §2; expected car ATE back to 0.31, total −0.0015. Clean val's car is the ruler for this (tv-series car 0.115–0.121 is directly comparable).

## 5. The last segment of the tail: the frame of the badly calibrated IMU
v20's public 0.3279 = the 0.3277 of the hand-picked coefficient override — the legal model has reached the level of a physical read-out. The next bottleneck is the attitude of the #29 IMU: AttNet corrects the 0037-type gyro frame only to 15°, and CalNet's 6-D head cannot learn a rotation from a 64-d summary (28.9°). A feasible alternative: a **differentiable Kabsch layer** — over the whole recording, use the relation between the time derivative du/dt of the low-passed accelerometer direction and −ω×u to solve directly for the constant rotation R_gyro→acc (the vector version of `calnet._tilt_consistency`), without network regression; validate on train against GT frames (the target file already has 134 recordings with resid < 8°). Gate: tilt < 5° on 0037-type recordings. If it passes, the four tail sequences could drop another 0.5–1.0.

## 6. Not to be done again
- Road 3 attitude synthesis (three cards 2.30–2.38 vs S 2.10, zero gain; S has already taught what it can).
- Road 1 stage 3 in its current form (frame-regression head).
- Ranking anything on a single-seed sprint fold.

## 7. Tonight
a3v22 official once (per the rules, swapping the second Kaggle entry); deliver a3v20_s42 (dry-run passed); add "three rulers are three populations" and the v20 seed range to the report. Tomorrow morning's budget is reserved for the two seeds of the §2 fix and the descriptor-fixed a3v22.
