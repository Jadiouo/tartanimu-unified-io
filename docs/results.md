# Results

The contest figures on this page can be checked against public sources (links in §6). The post-competition late-submission scores in §5 are visible only from my own Kaggle account.

**Terms.** *Private / public*: Kaggle splits the test data into two halves and reports a score for each (lower is better; predicting zero velocity everywhere scores 1.0). *AVE*: velocity error, metres per second. *ATE20*: trajectory position error, metres. *Sequence*: one test recording. *Final competition model* (code name `a3v20_s42`): the entry I delivered.

---

## 1. Final standing

**20th of 131 on the Kaggle private leaderboard; the organizers subsequently listed our team among the top 10 (organizers' e-mail, 2026-10-09); the official final ranking has not been announced.** IROS 2026 TartanIMU Challenge (CMU AirLab / Super Odometry), Kaggle team `Lexxxxx`, single human author (implementation by AI coding agents under my direction, see the README).
Private leaderboard score **0.19912**; public score of the final competition model 0.32793 (the leaderboard's public column shows 0.32695, the best public over all my submissions, which belongs to my second entry). The competition closed 2026-09-21; final standings were published 2026-09-28.

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

**Score** = 0.6·AVE/0.7356 + 0.4·ATE₂₀/3.1160, each macro-averaged over the four platforms; an all-zero submission scores 1.000. Of the 89 test sequences, 48 count for the private leaderboard and 41 for the public one; the organizers' scoring service uses all 89.

## 2. Official scoring service (all 89 test sequences)

For reference, the organizers' released 4-head baseline scores 0.937 on the public leaderboard when its best head is forced for every platform (an all-zero submission scores 1.054 there); it has no mechanism to infer the platform on the anonymised test, which is why this work trains one unified model instead.

This is the number I designed against. The scoring service returns per-platform and per-sequence errors, not only a total, so it is the honest estimator of the final standing; 33 queries were spent over the two weeks and every one is in [`../runs/official/`](../runs/official/).

| Model | Score ↓ | car ATE | human ATE | dog ATE | drone ATE / AVE | four-sprint drone tail, summed AVE |
|---|---|---|---|---|---|---|
| Earlier model: dense-window CNN + GRU, self-supervised pre-training, time-scaling augmentation | 0.2702 | 0.337 | 0.603 | 0.357 | 1.274 / 0.723 | 15.1 |
| + augmentation that speeds up drone-racing data in a drag-consistent way | 0.2658 | 0.313 | 0.639 | 0.357 | 1.244 / 0.703 | 12.7 |
| **+ learned sensor-calibration channels + gated velocity carry-over: final competition model (`a3v20_s42`)** | **0.2588** | 0.341 | 0.582 | 0.345 | 1.200 / 0.675 | 11.8 |
| second entry: no calibration channels, per-recording conditioning (`a3v21_s42`) | 0.2631 | 0.346 | 0.632 | 0.343 | 1.225 / 0.687 | 12.2 |
| competitor reference: AxisTilted2 (official 0.2156) | 0.2156 | 0.363 | 0.545 | 0.330 | 1.002 / 0.515 | — |

Delivered model, full breakdown: ATE₂₀ 0.61704 m, AVE 0.22017 m/s, RTE 0.95368; car 0.34105 / 0.0703, human 0.58245 / 0.05459, quadruped 0.34510 / 0.08106, drone 1.19955 / 0.67472.

Run-to-run variation of the delivered recipe, from changing only the random seed: 0.2588 (seed 42) / 0.2693 (seed 43). The pooled standard deviation (sd) across seeds is about 0.0056, so seed 42 is the favourable side and gaps under about 0.006 are not distinguishable. The two step gains in §3 (0.0044, 0.0070) are each below 2 sd and are not independent statistical evidence. Seed 42 was designated as the delivery seed **before** any seed was scored, and the second seed only reports the range.

## 3. What each step bought

| Step | Official Δ | Where the gain came from |
|---|---|---|
| dense windowing + cross-window GRU (day 1) | val 0.42 → 0.23 | context across windows, not capacity |
| self-supervised pre-training of the trunk (hide parts of the IMU signal, reconstruct them) | −0.014 val | 280× more targets than velocity supervision; stops trajectory memorisation |
| physically consistent time-scaling (`f' = k²(f − g·û) + g·û`), together with pre-training lengthened from 90 min to 3 h | 0.2837 → 0.2687 | fastest drone quintile −24 %; the two changes were not separated; time-warp augmentations of this family were used independently by at least four teams |
| drag-consistent speed-up augmentation (drone-racing data only) | 0.2702 → 0.2658 | the four sprint sequences: Σ AVE 15.1 → 12.7 m/s. All four sprints are in the public half, so this did not move the private rank |
| learned sensor-calibration channels (CalNet / AttNet) + gated velocity carry-over | 0.2658 → **0.2588** | tail 12.7 → 11.8, human ATE 0.639 → 0.582 |

All of it is drone: platform AVE went 0.844 → 0.675 between the first and the last official query, while car (0.31–0.34), human (0.58–0.65) and quadruped (0.33–0.39) ATE never moved structurally.

## 4. Honest notes

- **The private half compresses a lot of work into very few ranks.** Fourteen of my submissions land between 0.1932 and 0.2094 there — a band seven ranks wide (18th to 24th). The best of them (an earlier model, `a3dnTV3td`) would have been 18th; the delivered one is 20th. On the full 89-sequence test the delivered model is the best of my three finalist entries (0.2588 vs 0.2631 vs 0.2636), so on that estimator the rule's pick was the lowest of the three, though the margin (0.0043) is within seed noise and I do not claim it was the better model — the private half simply splits the four sprint sequences differently.
- **The gap to the top is drone, and mostly not the tail.** 84 % of the distance to the leading entries is drone AVE, and about 61 % of that sits on the 41 ordinary flights, the rest on the four sprints I spent the last four days on. The leading entry reaches drone AVE 0.515 with 64-second contexts and a smoother that tracks accelerometer bias and gravity — physics in a smoother, not in a forward recursion. Written up in [`peer_survey.md`](peer_survey.md).
- **What did not work is documented as carefully as what did**: about 50-60 entries (not counted line by line), each with the number that killed it, in [`chronicle.md §6`](chronicle.md).
- **Compliance**: one model, one checkpoint, no routing to per-robot sub-models, no ensembling, no test-time averaging, no external data in the delivered model, test labels never used. Discarded branches that touched external or unlabelled-test data were disclosed to the organizers and scored at most once; details in [`canonical_numbers.md §D`](canonical_numbers.md).

## 5. After the competition

Late submissions do not count for the ranking. Porting another team's published post-processing onto this network, then adding a ground-robot correction and per-window uncertainty smoothing, brought the Kaggle-returned scores to public 0.22342 / private 0.13903 (approx. full-test 0.178). I also submitted about 89 one-sequence probe files to recover which sequences are public and which private; the test values this revealed were not used to adjust any output. These are not contest results and are not placed in the table above; everything, with provenance, is in [`post_competition.md`](post_competition.md).

## 6. Verify

| What | Where |
|---|---|
| Public Kaggle profile, shows `TartanIMU Challenge … 20/131` | https://www.kaggle.com/lexhoooo/competitions |
| Private leaderboard, filtered to this team | https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026/leaderboard?search=Lexxxxx |
| Competition | https://www.kaggle.com/competitions/tartan-imu-challenge-iros2026 · https://superodometry.com/imuchallenge |
| Frozen weights + the exact scored `submission.csv` | https://huggingface.co/LexHo/tartanimu-a3v20 · https://huggingface.co/LexHo/tartanimu-a3v21 |
| Technical report submitted to the organizers | [`../report/technical_report.pdf`](../report/technical_report.pdf) |
| Screenshots, leaderboard exports, full submission history | [`evidence/`](evidence/) |

The delivery package re-executes from the frozen weights on CPU in 85 s in an isolated environment (no network, empty cache) and reproduces the scored `submission.csv` to a maximum component difference of 1.9 × 10⁻³ m/s.
