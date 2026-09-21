# Model-development playbook (distilled from the 13 TartanIMU days; 2026-09-21)

Purpose: the next time a "sensor sequence → regression" model is built, open the layer that matches the problem. Every layer has **symptom → what happened this time → rule → checklist**. Source: `chronicle.md`.

---

## 0. The ten problems that kept coming back (read this table first)

| # | Symptom | Instance this time | Root cause | Layer |
|---|---|---|---|---|
| 1 | The local metric does not move (or moves the wrong way) while the leaderboard moves, or vice versa | val is blind to groups A/B/C; public and full test disagree (a3sslc); a3v10 −0.072 on the fold, +0.018 official | each ruler measures a different population | L1 rulers |
| 2 | A long string of "no difference" | 30+ nulls 9/8–9/16; ten cards with zero signal on 9/16 | confounds not separated (several trajectories per batch, early stopping), or the ruler cannot see the effect | L1, L3 |
| 3 | Training loss keeps falling, val does not move | 60 windows, continuous trunk, experts, 3× oversampling | memorising trajectory fingerprints; velocity supervision is 3 numbers per window | L3 |
| 4 | Channels / sub-modules each pass their gate, the main network does not win | v_DR, AttNet, CalNet all pass; a3v19 loses to S alone | normalisation scale (std 3000) hides the channel from the trunk; the gate closes by epoch 30 | L4 |
| 5 | Wins on the fold, loses in-distribution, wins officially — all three at once | E1: sprint fold 1.89, val 0.1913, a3v20 official 0.2588 | three populations + one knob with one trade-off | L1, L5 |
| 6 | Augmentation helps the body and hurts the tail | T gives the 41 ordinary flights −16 % and the sprints +0.5–1.0 | the augmentation fixes the wrong physical ratio (k² vs k) | L5 |
| 7 | A cloud card has no artefact / crashed with no visible cause | path 2 six cards produced nothing; a3v22 crashed (missing moe.py); logs vanished with /workspace | bootstrap incomplete; logs not synced to persistent storage | L6 |
| 8 | Self-stop does not fire, or fires early | a 4-h budget would have killed a3v20_s43 9 min short; ss_lane processes disappeared | timer instead of completion condition; `pkill -f` kills its own shell | L6 |
| 9 | Rules / disclosure checked after the fact | test-IMU SSL, NeuroBEM, UZH-FPV were all done before learning they were not allowed | no "which data may enter the final" table before starting | L0 |
| 10 | Effort spent in the wrong place | the last four days all went to the tail (a fifth of the gap); four fifths is ordinary drone | "where is the gap" was not recomputed regularly | L2 |

---

## L0. Before starting (half a day)

**Rules and data boundaries**
- [ ] Snapshot the full rules page (`kaggle_rules_snapshot_*.md`) and list item by item: allowed / forbidden data, frozen weights, TTA/ensemble, re-run limits (GPU, memory, time, no network), deliverables, deadline (converted to local time).
- [ ] Write a "data availability table": each candidate source (released splits, external, synthetic, unlabelled test) → may it enter the final → does it need disclosure. **Without a written reply from the organisers, treat it as not allowed.**
- [ ] Measure the gap between the leaderboard top and the official baseline; decide first whether the aim is "win" or "explain a mechanism".

**Environment and verification**
- [ ] Data integrity (file counts, window counts, id set vs sample_submission).
- [ ] Local official scorer: feed the ground truth back → the score should be near the floor; all-zero → near the defined value. **No experiment before the scorer is verified.**
- [ ] Cache layer (`TARTANIMU_CACHE`, `CACHE_SCHEMA` version): bump the version whenever the data format changes, so a stale cache cannot be consumed silently.
- [ ] Submit all-zero once to confirm the submission pipeline.

---

## L1. Rulers (before methods; the most expensive lesson this time)

**Rule 1: state which population a ruler measures before reading it.** One change can "win, lose, win" on three rulers with all three correct.

**Order of building rulers**
1. Released val + official scorer (the daily in-distribution ruler; estimate the seed sd once, e.g. 0.002).
2. As soon as the **official breakdown / per-sequence table** is available, do "where is the error" (L2); find the populations val cannot see.
3. Build a **fold** for every population val cannot see:
   - Pairs from the same physical source (two IMUs on one flight, one session) are held out together; otherwise it is leakage.
   - The fold's difficulty is decided by "what remains in training": hold out all sprints = a pure extrapolation ruler (a channel learns the wrong sign); hold out one pair and keep the others = a tail ruler. **The fold design decides what a channel can learn.**
   - Measure the fold's noise first: same-seed reruns, cross-machine (this time sd 0.02–0.04, cross-machine offset 0.06) → each machine carries its own baseline.
4. Use **gates**, not **rankings**: threshold = 2× that ruler's seed sd; a candidate must not regress on any of the three gates and must pass at least one before an official query is spent.
5. The official full test only receives **pre-frozen** candidates; record the md5 of every query; plan the budget on the calendar.

**Decision protocol**
- Paired bootstrap (trajectory level, stratified by platform) + multi-seed (pairwise sqrt(s²/n + s²/n)); the threshold is the combination.
- New conditions are screened with one seed; stop immediately if worse by more than the threshold; add a second seed only on a signal; ≥ 0.010 to change the base.
- **Report the per-recording distribution of the change, not the mean** (a3v10's fold gain sat entirely on two side-flying recordings).
- val→public transfer is 1:1 across seeds of one recipe and ~3× across architectures — a small architecture-level val gain is worth more on the board than it looks.

---

## L2. Where is the error (recompute every 2–3 days)

- [ ] Official per-platform table + per-sequence → each platform's share of the score (drone 72 % this time).
- [ ] Sort per-sequence → how much do the top few carry (#29/#56/#37/#38 = 43 % of drone AVE).
- [ ] Group the sequences (tail / high-rotation / ordinary) and note for each whether val represents it (val only had group D).
- [ ] Decompose the **gap**: score formula → AVE vs ATE → which platform → which group. **Hit where the gap is; recompute before every plan.** (The last four days went to the tail = a fifth of the gap.)
- [ ] Find the data-source structure: noise-floor fingerprint, frame conventions (Kabsch fit of R_ext), recording length, session pairing. The two drone sources and 16 frame signatures explained half of the strange behaviour.
- [ ] Label-free test diagnostics are legal and cheap: p90 of your own predictions, counterfactual probes (scale a channel ×0.5 / ×1.5 and see whether the prediction moves).

---

## L3. Baseline and the hypothesis loop

**Baseline hygiene (before any method)**
- [ ] Separate the confounds: trajectories per batch, early stopping vs OneCycle (early stopping off by default), chunk boundaries, padding treated as observations, whether EMA is actually evaluated. All three bugs this time were caught by an external code review — **have every major version reviewed by someone else**.
- [ ] Record the training-loss / val ratio; a growing ratio is memorisation, not progress.

**The loop (20–30 min per card)**
1. One-sentence hypothesis + a pre-written gate (which ruler, how much).
2. Single factor, stacked on the current base, one seed.
3. Passes → add a seed; fails → into the negative-results catalogue with the number, **no chasing** (a single-seed null is dropped).
4. Devlog at the end of each day: what was done, why, what went wrong, the first thing tomorrow. Numbers go into the experiments table.

**Priors from this time (usable directly next time)**
- Works: dense random-offset windows, cross-window bidirectional GRU, several tokens per window, slow-filter up decomposition channels, small Huber β, masked-reconstruction pre-training of the trunk, physics-consistent augmentation, train+val last-epoch weights.
- Almost certainly does not work: capacity, longer context, Transformer mixer, MoE, the smoothing family (EMA/SWA/soup/Lookahead), the regularisation family, differentiable ATE surrogates, per-frame supervision, hard equivariance (when the premise fails), TTA/TTT.
- The cure for memorisation is "change the nature of the supervision" (reconstruction), not "add noise".

---

## L4. Putting physics / sub-modules into the network

- [ ] Every piece of physics passes its own oracle gate first (CalNet with GT-attitude integration, AttNet with held-out up error), **then** it is connected to the main network.
- [ ] The first thing after connecting: **normalisation scale** — print mean/std per channel; drift-type quantities must be bounded (q·tanh) with a fixed scale, or the trunk never sees them.
- [ ] Gates / gains print diagnostics: opening, fraction of trusted windows, pbetter, **split by population** (inside/outside the horizon, per platform) — an all-platform average is diluted by car/human (0.8 % vs. genuinely useful on the sprints).
- [ ] Rotations in 6-D, not rotation vectors; frame consistency under augmentation (yaw conjugation) has a numerical check (1e-6).
- [ ] Cache keys by content md5, not filename.
- [ ] The structural conclusion: **physics inside a regressor helps only where its assumptions hold, and a learned gate cannot recover the assumptions from the signal.** Where the assumptions (rest anchor, calibration, frame) fail for a population, the channel is noise there.

---

## L5. Augmentation and data coverage

- [ ] Write the augmentation as a physical formula and unit-test it (k=1 restores, gravity unchanged, labels scaled in sync).
- [ ] An augmentation fixes a "ratio" (T fixes drag/speed = k, S fixes it = 1); ask first which direction the test needs, then check per sequence whether anything was pulled the wrong way.
- [ ] Measure the acceptance rate of an augmentation per source (the "short recordings never get k > 1" bug would not show for a month).
- [ ] Criterion for external / synthetic data: compare a train+val candidate against the v6 seed band directly; a train-only twin is not enough (external data merely substitutes for val).
- [ ] External data: leakage check (cross-correlation) before use; competitions usually rule it out in the end — treat it as research, not delivery.

---

## L6. Cloud and automation (engineering rules, all learned the hard way)

- Separate persistent storage from the workspace (/vault vs /workspace); **sync logs to persistent storage every minute**; the saver packs succeeded and failed cards alike.
- The bootstrap is one script that installs **every** file (list generated from `git ls-files unified/`, not typed); print key-file md5s and a flag grep before launching.
- The card list can be appended while running; the runner reads by line index → insert only after the current line.
- Self-stop: **stop on completion** (every list entry has an artefact or FAILED) + a ceiling as insurance; when killing an old self-stop, filter yourself out with `pgrep`, never `pkill -f` (it kills the exec'ing shell).
- Each machine carries its own baseline; cross-machine offsets reach 0.06.
- Local CPU smoke test (2 epochs × 3 steps + predict.py consistency) before going to the cloud; a bug caught by the smoke test is ten times cheaper than a cloud card.
- Collector: the local machine polls the vault → pulls cards → runs the rulers automatically → writes a log; monitor only the log.
- Cost / time estimates: use measurements (5090: 20–30 min per card), not guesses; when the estimate is wrong, say so and update it.

---

## L7. Delivery (competition or deployment)

- [ ] Delivery package = frozen weights (sub-modules in the same file) + predict.py (split-only loader, no network) + pinned requirements + submission.csv (= the scored file, md5 recorded) + README (five compliance answers) + SHA256SUMS.
- [ ] Isolated dry-run: `env -i`, sockets blocked, empty cache, CPU; component-wise difference to submission.csv (tolerance 1e-3); record wall time / RSS.
- [ ] After uploading, **re-download from the remote into a clean directory and dry-run again**.
- [ ] One submission ↔ one repo ↔ one set of weights; README / form / report use one canonical numbers table (`canonical_numbers.md`).
- [ ] Register the selection rule in advance (seed pre-designated); count feedback usage honestly.
- [ ] Report structure: investigation records (what was measured → result → decision → what differed from expectation) separated from interpretation; negative results as detailed as positive ones.

---

## L8. Separating execution from review (worked, worth keeping)

- Keep the execution role (training, GPUs, delivery) and the review role (code review, analysis, recommendations) separate; communicate through a written log where every message carries numbers and tags.
- The review side does a code review before cards are spent (three rounds caught six bugs this time); the execution side lists "where the three rulers contradict each other" after every batch.
- The project owner decides only: goals, resource ceilings, approval of anything outside the plan, trade-offs that cannot be resolved; and the schedule.

---

## One-page decision tree (enter here when stuck)

```
Score not moving?
├─ local ruler moves, board does not → L1: the ruler's population ≠ the score's population → build a fold / read per-sequence
├─ training loss falls, val does not → L3: memorisation → change the supervision (reconstruction pre-training), stop the capacity line
├─ sub-module passes, main network does not win → L4: print channel std / gate diagnostics (split by population)
├─ fold wins, val loses → L5/L4: one knob, one trade-off; choose the population, or find an input that separates the two
└─ everything null → L2: recompute where the gap is; probably the wrong target

Card has no artefact / crashed? → L6: read the vault log first; bootstrap missing a file; flag smoke test
Time estimate wrong? → re-plan from measured card times, admit the error
Rules doubt? → L0: no written reply = not allowed; treat it as research, not delivery
```
