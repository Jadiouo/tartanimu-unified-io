# Results

Everything on this page can be checked against a public source; the links are in §5.

---

## 1. Final standing

**20th of 131 teams**, IROS 2026 TartanIMU Challenge (CMU AirLab / Super Odometry), Kaggle team `Lexxxxx`, single author.
Private leaderboard score **0.19912**; public leaderboard 0.32695 (18th). The competition closed 2026-09-21; final standings were published 2026-09-28.

| # | Team | Private score ↓ |
|---:|---|---|
| 1 | Marco | 0.12340 |
| 2 | Team - SPARO | 0.14733 |
| 3 | MobilityAI | 0.15088 |
| 4 | Haozhe Zhou | 0.15694 |
| 5 | AxisTilted2 | 0.15720 |
| … | | |
| 18 | Shake | 0.19633 |
| 19 | seewherewego | 0.19677 |
| **20** | **Lexxxxx** (this work) | **0.19912** |
| 21 | Dehan Shen | 0.20181 |

Full table: [`evidence/private_leaderboard.csv`](evidence/private_leaderboard.csv).

**Score** = 0.6·AVE/0.7356 + 0.4·ATE₂₀/3.1160, each macro-averaged over the four platforms; an all-zero submission scores 1.000. The private leaderboard uses about half of the test recordings, the public leaderboard the other 30 %, and the organisers' scoring service all 89.

## 2. Official scoring service (all 89 test sequences)

The number I designed against. The scoring service returns per-platform and per-sequence errors, not only a total, so it is the honest estimator of the final standing; 33 queries were spent over the 13 days and every one is in [`../runs/official/`](../runs/official/).

| Model | Score ↓ | car ATE | human ATE | dog ATE | drone ATE / AVE | sprint tail Σ AVE |
|---|---|---|---|---|---|---|
| official baseline (4 heads, best forced head) | — | | | | | public 0.937 |
| v6: dense-window CNN + biGRU, SSL trunk, time dilation | 0.2702 | 0.337 | 0.603 | 0.357 | 1.274 / 0.723 | 15.1 |
| + S-fast (drag-consistent scaling of the racing family) | 0.2658 | 0.313 | 0.639 | 0.357 | 1.244 / 0.703 | 12.7 |
| **+ learned inertial channels + gated recursion — delivered, `a3v20_s42`** | **0.2588** | 0.341 | 0.582 | 0.345 | 1.200 / 0.675 | 11.8 |
| variant B: no inertial channels, recording-level FiLM (`a3v21_s42`) | 0.2631 | 0.346 | 0.632 | 0.343 | 1.225 / 0.687 | 12.2 |
| competitor reference: AxisTilted2 (official 0.2156) | 0.2156 | 0.363 | 0.545 | 0.330 | 1.002 / 0.515 | — |

Delivered model, full breakdown: ATE₂₀ 0.61704 m, AVE 0.22017 m/s, RTE 0.95368; car 0.34105 / 0.0703, human 0.58245 / 0.05459, quadruped 0.34510 / 0.08106, drone 1.19955 / 0.67472.

Seed range of the delivered recipe: 0.2588 (seed 42) / 0.2693 (seed 43). Seed 42 was designated as the delivery seed **before** any seed was scored, and the second seed only reports the range.

## 3. What each step bought

| Step | Official Δ | Where the gain came from |
|---|---|---|
| dense windowing + cross-window biGRU (day 1) | val 0.42 → 0.23 | context across windows, not capacity |
| masked-IMU self-supervised pre-training of the trunk | −0.014 val | 280× more targets than velocity supervision; stops trajectory memorisation |
| physics-consistent time dilation (`f' = k²(f − g·û) + g·û`) | 0.2837 → 0.2687 | fastest drone quintile −24 % |
| S-fast (drag-consistent translation scaling, racing family only) | 0.2702 → 0.2658 | the four sprint sequences: Σ AVE 15.1 → 12.7 m/s |
| learned inertial channels (CalNet / AttNet) + gated recursion | 0.2658 → **0.2588** | tail 12.7 → 11.8, human ATE 0.639 → 0.582 |

All of it is drone: platform AVE went 0.844 → 0.675 between the first and the last official query, while car (0.31–0.34), human (0.58–0.65) and quadruped (0.33–0.39) ATE never moved structurally.

## 4. Honest notes

- **The private half compresses a lot of work into very few ranks.** Fourteen of my submissions land between 0.1932 and 0.2094 there — a band seven ranks wide (18th to 24th). The best of them (`a3dnTV3td`, day 7) would have been 18th; the delivered one is 20th. On the full 89-sequence test the delivered model is the best of the three finalists (0.2588 vs 0.2631 vs 0.2636), so the pre-registered rule picked correctly against the better estimator — the private half simply splits the four sprint sequences differently.
- **The gap to the top is drone, and mostly not the tail.** 84 % of the distance to the leading entries is drone AVE, and four fifths of that sits on the 41 ordinary flights, not on the sprints I spent the last four days on. The leading entry reaches drone AVE 0.515 with 64-s contexts and a Gaussian smoother carrying accelerometer-bias and gravity states — physics in a smoother, not in a forward recursion. Written up in [`peer_survey.md`](peer_survey.md).
- **What did not work is documented as carefully as what did**: ~60 interventions with the number that killed each one, in [`chronicle.md §6`](chronicle.md).
- **Compliance**: one model, one checkpoint, no platform routing, no ensembling, no TTA, no external data in the delivered model, test labels never used. Discarded branches that touched external or unlabelled-test data were disclosed to the organisers and scored at most once; details in [`canonical_numbers.md §D`](canonical_numbers.md).

## 5. Verify

| What | Where |
|---|---|
| Public Kaggle profile, shows `TartanIMU Challenge … 20/131` | https://www.kaggle.com/lexhoooo/competitions |
| Private leaderboard, filtered to this team | https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026/leaderboard?search=Lexxxxx |
| Competition | https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026 · https://superodometry.com/imuchallenge |
| Frozen weights + the exact scored `submission.csv` | https://huggingface.co/LexHo/tartanimu-a3v20 · https://huggingface.co/LexHo/tartanimu-a3v21 |
| Technical report submitted to the organisers | [`../report/technical_report.pdf`](../report/technical_report.pdf) |
| Screenshots, leaderboard exports, full submission history | [`evidence/`](evidence/) |

The delivery package re-executes from the frozen weights on CPU in 85 s in an isolated environment (`env -i`, sockets blocked, empty cache) and reproduces the scored `submission.csv` to a maximum component difference of 1.9 × 10⁻³ m/s.
