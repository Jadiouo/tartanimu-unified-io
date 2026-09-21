# Night of 9/20 → early 9/21: the recursion-gain study (the `advice_after_v20` plan executed)

Scope: §2 EKF variants (main line), §3 FiLM descriptors, §4 dr_horizon mask, §5 Kabsch; plus the ablations that came out of the review (carry / rot / carryS), target variants (nosup / margin / mask_direct / nsmd), FiLM v1 seed twins and a3v22. No official query was spent (everything judged on the three sprint folds, fold A and clean-val twins).

## 1. Conclusions

1. **The recursion head (EKF) is one knob with one trade-off**: how far the gain opens = sprint-tail gain vs in-distribution lag. Gain probe (mean gain per platform on val, 1 = direct head):
   - ≈ 1.00 (nosup, nsmd) → val ≈ S alone (0.1798 / 0.1815), the sprint gain disappears;
   - 0.97 (a3v20 recipe) → tail −0.9 m/s, in-distribution +0.004, official 0.25878 (rewarded);
   - 0.60–0.88 (E1, margin) → a uniform +0.01 lag on all four platforms, val 0.188–0.191.
   The gain has no input that separates "a sprint next to a rest anchor" from "a car starting from rest". **The recursion line is closed; a3v20_s42 stays the delivery.**
2. **The physical increments really help the sprints** (it is not smoothing): on fold 1 E1 1.89, carry (dv=0) 2.24, rot (rotation only) 2.25, S alone 2.10. The held-out sprints and the four test-tail sequences all anchor at window 0, within the 10-s horizon, where the window-integrated increments carry information.
3. **The same increments hurt car/human in-distribution**: a car starts from rest → good anchor → increments open → errors in the first 10 s; mask_direct (masked windows use the direct head) does not recover it (val 0.1878) and only reaches 2.09 on the sprint fold (E1 1.89) — learning the gain only inside the horizon means the gain never learns "when to close", and the sprint gain goes with it.
4. **FiLM v1's val gain is seed-stable** (tv_film s42 0.1746 / s43 0.1770, S 0.1784); the yaw-invariant descriptor v2 is worse (0.1797, fold 2.23) — the body-frame means are what FiLM uses to recognise a recording. → Kaggle finals **a3v20_s42 + a3v21_s42**; **a3v22_s42 (a3v20 + FiLM v1) trained**, official on 9/21 morning.
5. Road 5 (Kabsch) has no information on the sprints (thrust-dominated); the sig-7 family is dominated by accelerometer bias and its GT is not a rigid rotation of the IMU frame → parked.

## 2. Three sprint folds (0037/39, 0038/40, 0041/42 held out; sprint mean / fold AVE)

| Recipe | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|---|
| S alone | 2.10 / 1.555 | 1.76 / 1.437 | 2.49 / 1.699 | **2.12** |
| a3v20 recipe (old EKF) | 2.29 / — | 1.66 / 1.352 | 2.27 / 1.722 | **2.07** |
| E1 (int increments + mask features + soft target + horizon 10) | 1.89 / 1.489 | 1.62 / 1.315 | 2.40 / 1.737 | **1.97** |
| carry (channels, dv=0, dR=I) | 2.24 / 1.606 | — | — | |
| rot (dv=0, dR gyro) | 2.25 / 1.651 | — | — | |
| carryS (14-channel pure smoothing) | 2.17 / 1.665 | 1.65 / 1.394 | — | |
| FiLM v2 | 2.23 / 1.711 | — | — | |
| E1 + FiLM v2 | 2.38 / 1.804 | — | — | |
| e1md (E1 + masked windows → direct head) | 2.09 / 1.556 | — | — | |

Fold A: E1 1.260 (S-fast 1.377, fa_ekf 1.228, fa_dr2 1.198).

## 3. Clean-val twins (official scorer; ATE20 / AVE)

| Twin | val | car | dog | drone | human | Gain probe |
|---|---|---|---|---|---|---|
| tv_sfast (S alone) | 0.1784 | .363/.089 | .397/.081 | .784/.317 | .509/.065 | — |
| tv_film s42 / s43 (FiLM v1) | 0.1746 / 0.1770 | .332/.089 / .330/.088 | .375/.080 / .376/.081 | .788/.308 / .822/.317 | .517/.064 / .495/.064 | — |
| tv_ekf (a3v20 recipe) | 0.1821 | .351/.093 | .374/.083 | .821/.329 | .498/.066 | .97 |
| tv_e1 | 0.1913 | .378/.098 | .404/.092 | .830/.332 | .579/.072 | — |
| tv_e1nosup | 0.1798 | .373/.091 | .327/.081 | .805/.318 | .567/.065 | 1.00 |
| tv_e1md (mask → direct head) | 0.1878 | .396/.098 | .400/.089 | .837/.324 | .547/.068 | — |
| tv_e1margin | 0.1887 | .379/.097 | .404/.090 | .844/.330 | .533/.069 | .60–.88 |
| tv_e1nsmd | 0.1815 | .348/.089 | .337/.082 | .823/.330 | .545/.066 | 1.00 |
| tv_carryS | 0.1828 | .364/.092 | .421/.081 | .827/.322 | .512/.067 | — |
| tv_carry (channels, dv=0, dR=I) | 0.1847 | .383/.095 | .357/.082 | .847/.332 | .506/.068 | — |
| tv_film2 (descriptor v2) | 0.1797 | .116* | .116* | .366* | .122* | — |
| tv_e1f (E1 + v2) | 0.1913 | .397/.097 | .404/.091 | .856/.330 | .576/.070 | — |

(* score column only.) Gate: per-platform ATE and AVE no worse than tv_sfast — no recursion card passes.

## 4. Diagnostics (split by population)
E1 family: vprop_p99 8.6 (old 15.3), pbetter 17 %, **pbetter_u 0.7–0.9 % / pbetter_m 17 %, mask_frac 12.7–13.4 %** — averaged over all platforms the physical propagation is barely more accurate than the direct head, but that average is diluted by the many car/human windows; on the sprint fold, removing the increments removes the gain (§1.2).

## 5. CPU follow-ups
- Anchors: even-IMU racing recordings < 1.3° (except 0038/0040, no rest window); odd-IMU recordings 7–9° because of accelerometer bias; consistency-based anchor selection is worse. The four test-tail sequences (#29/#56/#37/#38 = test_0028/0055/0036/0037) all anchor at window 0; the one without any rest is #57.
- sig-7 family: rate-level consistency says the gyro should be −I (0.12 vs R_MAP 1.0), but after integration all three maps give 6–8° (an accelerometer bias of ~2.5 m/s² dominates); the GT-attitude oracle itself gives 19 m/s → parked.
- The closed-form Kabsch solution has no information on the sprints (residual ~2).

## 6. Morning of 9/21 (as planned)
- Official: `a3v22_s42` (md5 79ee1f14…; label-free sensitivity: the four tail sequences' p90 is 0.2–0.8 m/s faster than a3v20). Condition to replace a3v20: official < 0.2588 and car ≤ 0.34, dog ≤ 0.36, human ≤ 0.65 ATE. (Outcome: 0.26359, car 0.355 → not replaced.)
- Kaggle finals (ticked on the site before 07:55): a3v20_s42 + a3v21_s42.
- Delivery: `hf_release_a3v20/` (dry-run passed).
- Report: the recursion line written up as a negative result (the knob figure of §1 + the gain probe + fold/val decomposition); the yaw inconsistency of FiLM v1 noted as tolerable (0.1746 / 0.1770).
