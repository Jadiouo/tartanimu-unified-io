# One unified inertial velocity model for four robots — IROS 2026 TartanIMU Challenge

One network, one set of weights, four platforms (car, quadruped, drone, handheld): raw 6-axis IMU in, body-frame velocity out. Official full-test **TartanIMU Score 0.25878** (ATE₂₀ 0.617 m, AVE 0.220 m/s), Kaggle team `Lexxxxx`, single author, 13 days.

This repository is the public, curated version of the work: the model and training code, the evaluation rulers, the cloud automation, the technical report, and — the part I think is most useful to others — a full record of what was measured, what was decided on it, and where the outcome differed from what I expected.

<p align="center"><img src="figures/pipeline.png" width="720" alt="pipeline"></p>

## Results (official per-sequence scoring service, all 89 test sequences)

| Model | Score ↓ | car ATE | human ATE | dog ATE | drone ATE / AVE | sprint tail Σ AVE |
|---|---|---|---|---|---|---|
| v6: dense-window CNN + biGRU, SSL trunk, time dilation | 0.2702 | 0.337 | 0.603 | 0.357 | 1.274 / 0.723 | 15.1 |
| + S-fast (drag-consistent scaling of the racing family) | 0.2658 | 0.313 | 0.639 | 0.357 | 1.244 / 0.703 | 12.7 |
| **+ learned inertial channels + gated recursion (delivered, `a3v20_s42`)** | **0.2588** | 0.341 | 0.582 | 0.345 | 1.200 / 0.675 | 11.8 |
| variant B: no inertial channels, recording-level FiLM (`a3v21_s42`) | 0.2631 | 0.346 | 0.632 | 0.343 | 1.225 / 0.687 | 12.2 |

Seed range of the delivered recipe: 0.2588 / 0.2693 (seeds 42 / 43; seed 42 was designated before any seed was scored). All 33 official queries with per-platform and tail breakdown: [`runs/official/`](runs/official/), summarised in [docs/chronicle.md §5](docs/chronicle.md).

Frozen weights and the exact prediction files: [huggingface.co/LexHo/tartanimu-a3v20](https://huggingface.co/LexHo/tartanimu-a3v20) · [tartanimu-a3v21](https://huggingface.co/LexHo/tartanimu-a3v21) (each re-executes on CPU in ~85 s, verified against `submission.csv` to 2e-3 m/s).

## What is in the model

- **Input**: dense random-offset 1-s windows; 25 channels = raw IMU (6) + decomposition about a slow complementary-filter "up" (8) + rest reference (4) + learned-INS channels (7).
- **Learned-INS channels** from two small sub-modules stored in the same checkpoint and frozen: `CalNet` (64-d recording summary → accelerometer scale / bias / time constant / gyro-axis map) and `AttNet` (rest-anchored SO(3) gyro integration with GRU corrections). From them: a bounded dead-reckoned velocity `q·tanh(∫(R f + g)/q)`, time since the rest anchor, and the rotation to the anchor frame.
- **Trunk / context / heads**: dilated 1-D ResNet (5 tokens per window) → bidirectional GRU over 40 windows + 5-s low-pass branch → a direct velocity head plus a learned-gain recursion `v_k = ΔR_k v_{k−1} + ΔV_k + K_k (v_direct − v_prop)` over the window-to-window IMU increments. 2.17 M parameters + 78 k in the sub-modules.
- **Training**: masked-IMU self-supervised pre-training of the trunk (released train+val IMU only), then 120 epochs on train+val, last-epoch weights. Augmentation: physics-consistent time dilation (`f' = k²(f − g·û) + g·û`), **S-fast** (`f' = s(f − g·û) + g·û, v' = s·v` on the identity-source racing family only), direction-only yaw augmentation.
- **Inference**: deterministic, offline within one trajectory, no platform label, no ensembling, no TTA, no adaptation. Only the released train/val splits were used.

## What I learned (the short version)

1. **Every ruler measures a population.** The released validation split is honest for ordinary drone flights and blind to the sprint tail that decides 43 % of the drone error; a high-rotation fold and a sprint fold measure two other populations. A change can win, lose and win on the three at once, and all three are right. I switched from ranking on rulers to gating on them (threshold = 2× the ruler's seed spread) and stopped reading the public leaderboard.
2. **Fold design decides what a channel can learn.** Holding out *all* sprints made a rest-reference channel learn the wrong sign; the same channel was null on a fold that keeps sibling sprints in training.
3. **The models did not read drag.** A counterfactual probe (scale the horizontal specific force of the two sprint sequences by 0.5 / 1.5) moved predictions by < 0.3 m/s. Time dilation *raises* the sprints (drag ∝ k², speed ∝ k); S-fast fixes the ratio at 1 and is the first of ~30 interventions that moved the tail.
4. **Physics inside a regressor helps only where its assumptions hold, and a learned gate cannot recover the assumptions from the signal.** The recursion gain is one knob with one trade-off: gain ≈ 1 reproduces the no-physics model, 3 % propagation gives the delivered gain, 12–40 % gives a uniform in-distribution lag on every platform. 22 ablations in one night, all documented as a negative result.
5. **Sensor conventions vary inside one platform.** 239 of 282 training drone recordings have an IMU frame that is "upside down plus a recording-specific yaw offset" relative to the labels; random body-yaw augmentation destroys the other source. The strongest public entries (official 0.216–0.238) win on the ordinary drone flights with 64-s contexts and a bias/gravity-state smoother, not on the tail.

Full record: [docs/chronicle.md](docs/chronicle.md) (day-by-day timeline, architecture evolution, all official queries, negative-results catalogue, index) · [report/technical_report.pdf](report/technical_report.pdf) (English, IROS workshop template: eight investigations with *measured / result / decision / what differed*) · [docs/workflow_playbook.md](docs/workflow_playbook.md) (the development workflow distilled from the recurring problems) · [docs/recursion_gain_study.md](docs/recursion_gain_study.md) · [docs/peer_survey.md](docs/peer_survey.md) (what the other public entries did).

## Repository layout

```
predict.py, requirements.txt    inference entry point (identical to the HF repos)
unified/                        model, data pipeline, augmentation, sub-modules, training scripts
local_eval/                     rulers: official scorer on val, sprint/rotation folds, label-free probes
runs/official/*.csv             every official scoring-service result (overall / platform / per-sequence)
runs/folds/*.json               the stress-fold definitions
cloud/                          GPUtw automation: bootstrap, lane runner, self-stop, result collector
report/                         LaTeX source + PDF of the technical report
docs/                           chronicle, playbook, peer survey, recursion-gain study, canonical numbers
```

Data are not redistributed (competition licence: academic research only). Cite Tartan IMU (Zhao et al., CVPR 2025) and the [challenge page](https://superodometry.com/imuchallenge/) if you build on this.

## Reproduce a submission

```bash
pip install -r requirements.txt
# download weights/tartanimu_a3v20_s42.pt from the HF repo into weights/
python predict.py --data /path/to/tartan-imu-challenge-iros2026 --out submission.csv
```

Training flags of the delivered model are stored in the checkpoint's `args`; the recipe line is in `docs/canonical_numbers.md`.

## Acknowledgements

CMU AirLab / Super Odometry Group for the dataset, the scoring service and the rule clarifications; GPUtw for cloud GPUs. Code review and analysis were assisted by AI tools; all experiments, numbers and claims were verified by the author.

## Licence

Code and documents: MIT. Weights: see the HF repositories. Dataset: competition licence, not included.
