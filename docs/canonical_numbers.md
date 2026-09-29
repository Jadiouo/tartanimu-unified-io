# Canonical numbers (2026-09-21 02:30) — shared by README / submission form / technical report / HF repos

Every number and wording follows this file; change here first, then sync the other three.

## A. Deliverables

| Item | Value |
|---|---|
| Team (Kaggle) | `Lexxxxx` (single author) |
| **Final standing (published 2026-09-28)** | **20th / 131 teams**, private leaderboard **0.19912**; public leaderboard 18th, 0.32695 |
| Main submission (Kaggle final #1, HF, report subject) | `a3v20_s42`, CSV `sub_test_a3v20_s42.csv`, md5 `545d7cddc859a16db045f872251be03c` |
| Official full test (scoring service) | **Score 0.25878**, macro ATE20 0.61704, macro AVE 0.22017, RTE 0.95368 |
| Official per-platform ATE20 / AVE / RTE | car 0.34105 / 0.0703 / 0.26476; human 0.58245 / 0.05459 / 0.30122; quadruped 0.3451 / 0.08106 / 0.3749; drone 1.19955 / 0.67472 / 2.87383 |
| Kaggle public | 0.32793 (submitted 2026-09-20 20:11) |
| Second submission (Kaggle final #2; own HF repo; the PDF is attached only to the last form) | `a3v21_s42`, md5 `a53e196f6b67e8ea52733f4386b5c403`, official 0.26306 (ATE20 0.63665, AVE 0.22232), public 0.32695; car 0.346 / human 0.632 / dog 0.343 / drone 1.225 / 0.687 |
| Weights | `weights/tartanimu_a3v20_s42.pt`, sha256 `071994ba24bbe45648dc010f4c899f38f4b62721d092a8904efa0dad1b1eeabb` (one file, containing the `attnet` and `calnet` sub-modules) |
| HF repo (main) | https://huggingface.co/LexHo/tartanimu-a3v20 (public, commit 2426e014, 31 files) |
| HF repo (second) | https://huggingface.co/LexHo/tartanimu-a3v21 (public, commit 1b0d0829, 31 files; weights sha256 7d90bade…; SHA256SUMS pass after download; isolated dry-run 69 s / 1.4 GB / max diff 1.6e-3) |
| Re-run verification | re-downloaded from HF, isolated environment (env -i, sockets blocked, empty cache, CPU): 85 s, 2.0 GB RSS; component-wise difference to submission.csv mean 2.0e-5 / p99 1.7e-4 / max 1.9e-3 m/s; SHA256SUMS pass |
| Environment | torch 2.12.0, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1; 1 GPU < 1 min < 1 GB VRAM |

## B. Method in one paragraph (form "Method Summary")

Single unified model, one shared set of weights for all four platforms. Two small learned sub-modules stored in the same checkpoint pre-process each recording's raw IMU (CalNet: accelerometer scale / bias / time constant and gyro-axis map; AttNet: rest-anchored SO(3) gyro integration with GRU corrections), from which we derive a bounded dead-reckoned velocity, time since the rest anchor and the rotation to the anchor frame. Main network: dense random-offset 1-s windows, 25 input channels (raw 6-axis IMU + 8-channel decomposition about a slow complementary-filter up direction + the learned-INS channels) → dilated 1-D ResNet trunk (5 tokens/window) → bidirectional GRU over 40-window chunks → a direct velocity head plus a learned-gain recursion v_k = ΔR_k v_{k−1} + ΔV_k + K_k (v_direct − v_prop) inside the same network → body-frame velocity. Huber loss. Trunk initialised by masked-IMU self-supervised pre-training on the released train+val IMU only. Trained on train+val, 120 epochs, last-epoch weights, seed 42 pre-designated. No TTA, no ensembling, no platform routing, no external data; inference from raw IMU only, one deterministic pipeline per recording.

## C. What is novel (form "What is novel")

(1) "S-fast": a drag-consistent translation-scaling augmentation (s ∈ [1, 1.8]) applied jointly to the velocity labels and the gravity-removed specific force of the identity-source racing-quadrotor family, together with time dilation (k ∈ [0.7, 1.5]); it is what moves the fast-sprint tail of the test set (four straight-sprint sequences: summed AVE 15.1 → 12.7 m/s vs. the same recipe without it). (2) Learned-INS input channels from two per-recording sub-modules (calibration network, attitude network) and a gated recursion head that carries a physics-propagated velocity between windows; on top of S-fast it lowers the tail further (12.7 → 11.8) and human / quadruped ATE (0.639 → 0.582, 0.357 → 0.345), giving the official 0.25878 (v6 baseline 0.2702, S-fast alone 0.26577). (3) Honest characterisation: the gain of the recursion is one knob with one trade-off — propagation helps sprints within ~10 s of a rest anchor and hurts car / human in-distribution; the delivered model uses ≈3 % propagation.

## D. Five compliance answers (same text in README and report §6)
1. One model, one set of weights: yes — one checkpoint file (main network + two frozen pre-processing sub-modules), one deterministic pipeline per recording / chunk; the sub-modules only produce input channels and increments, the only velocity output is the main network's (no second estimator, no blending).
2. No platform classification / routing / experts at inference; the auxiliary platform-logit head is a training regulariser and is discarded at inference; CalNet's gyro-map class is an IMU mounting correction (acting on the input signal), not a platform switch.
3. No ensemble, no checkpoint averaging, no TTA; overlapping chunks of the same model's own predictions are averaged (stride K/2).
4. Test labels never used; the final checkpoint = the fixed last epoch; SSL used train+val IMU only; no external data / external pre-trained weights entered the delivered model. Disclosure: on 9/12–13 one discarded early branch was pre-trained on train+val+test raw IMU, scored once, excluded the next day, no derived weights; the NeuroBEM external data were used on 9/18 in supervised fine-tuning experiments and scored, and per the organisers' written ruling are not a final candidate; UZH-FPV was deleted unused once it was found to overlap the competition recordings; a physics post-processing probe (override) was scored once on the public LB and is not part of any submission. Feedback disclosure: 33 official full-test queries (distinct tags in runs/official), 26 Kaggle submissions (to 9/21 02:00).
5. Development vs delivery: the same inference code; the delivered checkpoint differs from early LB entries by the training augmentation (S-fast) and the learned inertial channels / recursion head; model selection used the released val and two training-set stress folds.

## E. Report elements (outline)
1. Summary (numbers of A)
2. Method (B expanded: input channels; CalNet / AttNet; derived channels; trunk / GRU / direct head + recursion head; SSL; augmentation; training)
3. Results: official Tables III–V; ablation chain v6 0.2702 → a3v16S 0.26577 → a3v20 0.25878; seed ranges (a3v20 s43 0.26933; S-fast s43 0.27336; v6 four seeds 0.270–0.281); a3v21 (FiLM) 0.26306, a3v22 (a3v20 + FiLM) 0.26359
4. "Three rulers are three populations": sprint fold (groups A/B), fold A (group C), clean val (group D + the other platforms); gates not rankings; three-fold sprint table; val-twin table; gain-probe figure (≈1.0 / 0.97 / 0.6–0.88)
5. What did not help (incl. the 9/20-night recursion variants, descriptor v2, Kabsch, sig-7 family analysis, roads 2 / 3)
6. Error analysis (drone 70 %, tail coefficient, #57 without rest)
7. Compliance (D)
8. Reproducibility (the re-run row of A)
9. Compute

## F. Known inconsistencies to fix
- ~~predict.py docstring~~ fixed and pushed (commit 2426e014).
- ~~README re-run numbers~~ changed to the HF-download figures 85 s / 2.0 GB (commit 2426e014).
- Report v3 §6 feedback counts [fill]: the official query count and Kaggle submission count are counted from `runs/official/` and the Kaggle API.
