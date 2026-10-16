# After the competition (2026-10-10 to 10-16)

**Everything on this page is post-competition work.** Late submissions on Kaggle do not count for the ranking. The contest result is: 20th of 131 on the Kaggle private leaderboard; the organizers subsequently listed our team among the top 10 (organizers' e-mail, 2026-10-09); the official final ranking has not been announced. The private score is 0.19912 and the official full-test score is 0.25878 (my final competition model, code name `a3v20`). The numbers below must not be read as a continuation of that rank or drawn on one "improvement line" with it.

**Terms used on this page.**
- *Public / private*: Kaggle splits the test data into two halves (41 and 48 sequences) and reports a score for each. Lower is better; an all-zero prediction scores 1.0.
- *Full-test score*: the organizers' own scoring service over all 89 test sequences. It is not available after the contest, so after the contest I estimate it from the public and private scores (conversion explained below).
- *Held-out set*: recordings taken out of the training data entirely, used as an exam the model has never seen.
- *Pass/fail rule written first* (I call these "pre-registrations" in the file names): before each submission or check I wrote down what result would count as success.
- *thisray*: another team, which released a post-processing method after the challenge.

## 1. What happened, in one paragraph

After the contest I took the physics-based post-processing method that thisray (Ssu-Rui Lee) had published on Hugging Face and ran it on top of my own network's output (**Step 1**). That method is not mine. Its code is Apache-2.0 and is called unmodified, with only a data interface written by me; its 267-parameter readout weights are CC-BY-NC-4.0 (research use). On top of it I added two small steps of my own: a correction for ground robots applied at half strength (**Step 2**) and a per-window uncertainty weight inside their smoothing stage (**Step 3**). Each of my two steps was first checked on recordings held out of training and submitted against a pass/fail rule written down beforehand. For Step 1 the evidence was weaker, as section 3 says.

## 2. Stage table

Public = 41 test sequences, private = 48; both are exact values returned by Kaggle for late submissions. "Approx. full test" is my conversion `0.4632·public + 0.5368·private` (the weight is back-solved from the final competition model; two sub-steps that change only drone sequences use a drone identity instead). The conversion is not a re-score by the organizers. Checks on three entries whose official scores I know gave errors of -0.00038, +0.00034 and +0.00048, so read it as about ±0.0005, and possibly worse for the later rows, which touch every platform.

Step 1 was built up over four late submissions, which keep their internal letters (A, C, D, E) in this table so the files in the repository can be matched. A = thisray's drag-based drone smoother on 9 drone sequences; C = the same on two more sequences; D = their ground-robot and drone-flight operators added; E = their small readout model added, which completes their full public chain.

| Step | What was added | Whose | Public | Private | Approx. full test |
|---|---|---|---|---|---|
| Final competition model (reference) | none | mine | 0.32793 | 0.19912 | 0.25878 (exact, official) |
| 1, sub-step A | drone drag smoother on 9 drone sequences | borrowed (thisray) | 0.30870 | 0.17867 | ~0.2390 |
| 1, sub-step C | + 2 more drone sequences (one flight) | borrowed; selection rule mine | 0.26277 | 0.17867 | 0.21548 (drone identity) |
| 1, sub-step D | + operators for drone flight, dog, car, human | borrowed (thisray) | 0.23139 | 0.14099 | ~0.1829 |
| **1, sub-step E (Step 1 complete)** | + readout (= my network + thisray's full public chain) | borrowed (thisray) | 0.22671 | 0.14372 | ~0.1822 |
| **2** | + half-strength correction on detected ground trajectories | mine | 0.22468 | 0.13957 | ~0.1790 |
| **3** | + per-window uncertainty weight in the smoother | mine (the smoother is thisray's) | 0.22342 | 0.13903 | ~0.1781 |

![post-competition stages](../figures/post_competition_stages.png)

How to read it:
- Relative to my final competition model, public and private are both lower at every step. Between sub-steps it is not monotone: from D to E, private rose from 0.14099 to 0.14372 while public fell.
- Approximate increments: D to E -0.0007; E to Step 2 -0.0032; Step 2 to Step 3 -0.0009; E to Step 3 -0.0040. The Step 2 to Step 3 change is the size of the conversion error and shows a direction only. E to Step 2 is about six times the error but is still approximate.
- Most of the gain from the competition model to Step 1 comes from the borrowed method. The part that is mine is the change from E to Step 3, about -0.004 approximately.
- Against thisray's own reported 0.18775 (single run, from their report, not re-verified by me), Step 1 at about 0.182 is of the same order. The difference of about 0.006 is within the run-to-run variation caused by random seeds (pooled standard deviation about 0.0056), so I do not claim to be better or worse. Their network alone is reported at 0.27222, mine at 0.259 / 0.269 (seeds 42 / 43), again the same order. My network has 2.17 M parameters and theirs 43.7 M, about 20 times as many.
- None of these late scores has a rank; they must not be placed in the contest table.

## 3. What each step did, who did it, and how it was checked

**Step 1, sub-steps A and C (drag-based smoother for drones).** thisray's smoother uses a physics model of rotor drag, applied to the drone sequences their own rule assigns to it. A: 9 sequences. C: two more sequences from one flight, included after A under a one-sided rule (include if the approximate full-test gain is at least 0.002). Pass criterion for A, written before submission: approximate official change at or below -0.010 (extrapolation had expected -0.005); result -0.0198. Both public and private moved, so it was not one or two sequences in one half. C's whole effect landed in public (private unchanged) and comes from one flight; choosing to include those two sequences uses one bit of test feedback.

**A clean check for the drone operator.** A model trained without 12 drone recordings; the operator was checked on those.

**Step 1, sub-steps D and E (thisray's operators for drone flight, dog, car and human, then the readout).** Held-out check before submitting: the drone-flight operator reduced 4 held-out recordings from 0.157 to 0.137; the readout lowered the 12-recording average by 20% but hurt the recordings the flight operator handles (0.16 to 0.26), so it is not applied to them. D and E were both submitted and the better one kept: one extra pick between two options. For car, dog and human there was no clean positive evidence at that point (on the in-sample validation set, my final competition model got worse everywhere after the chain, which is uninformative because that model had trained on it). That gap is what the next steps address.

**Finding on ground robots.** On 27 held-out ground recordings (car 11, dog 9, human 7), the full chain barely changes my network's output on net (score +0.19% / +1.01% / -0.42% across three trained models); on drones it reduces velocity error by 49.0% (12 held-out recordings) and on car it makes velocity error worse (+14.6%). The correction is good where its assumptions hold (aerodynamic drag) and neutral to harmful elsewhere.

**Step 2: ground correction at half strength (mine).** On trajectories detected as ground (car 19, dog 14, human 10 in the test set) the output is `prior + 0.5·(chain − prior)`, where "prior" is my network's own output and "chain" is the output after thisray's full chain. The idea of shrinking the chain on ground came from Codex on 2026-10-13. The coefficient 0.5 was chosen after the fact on the 27 held-out recordings, and the ground-detector threshold was fitted on training data that includes those recordings, so the checks below are not fully independent.

| Check (27 held-out ground recordings) | Result |
|---|---|
| Replication run 1 (old checkpoint) | score -5.74% vs the full chain, 95% interval excludes 0 |
| Replication run 2 (seed 42 retrained) | -6.12%, interval excludes 0 |
| Replication run 3 (seed 43) | -5.67%, interval excludes 0 |
| Without the car segment | -2.25% / -2.85% / -1.28%, intervals cross 0, so not claimed |
| Independent recomputation (Codex) | numbers match to < 1e-6; interval [-0.0121, -0.0043] |

Kaggle result: both public and private improved as pre-specified; only the 43 ground trajectories changed, and the drone output is unchanged from Step 1.

**Step 3: per-window uncertainty in the smoother (mine).** My idea: feed a per-window uncertainty into the smoother's weights, so it trusts confident windows more. The uncertainty estimate is how much several predictions of the same window disagree with each other. On the 27 held-out ground recordings the score changed by -0.00083 to -0.00094 with intervals excluding 0, and a control with the uncertainty values shuffled got worse in 8 of 9 cases. Kaggle result: both improved by a small amount; about -0.0009 approximate, comparable to the conversion error.

## 4. Negative results in this phase

- **A network that predicts the uncertainty directly** (instead of using disagreement between predictions): ground score change -0.00041, about half of the disagreement-based gain; the drone arm failed (+0.0095, interval spans 0, 1 win / 6 losses / 5 ties). Not submitted.
- **Averaging predictions over several alignments at test time:** my idea. All four comparison intervals cross 0 on 12 drone + 27 ground held-out recordings; not adopted. The sign was mostly favourable but effects were around 1e-4, far below noise. One deviation from the written "stop if checks fail" rule: two checkpoints differed by up to 1.6e-4 from their stored CPU/GPU outputs (a floating-point difference, mean 1e-7); I proceeded and noted it.
- **New ground-robot constraints** (four candidates, e.g. a car no-slip constraint): none met the pass criterion; the car constraint made car position error worse by 36 to 44%.
- An earlier line of this analysis (an "additive" path that predicted 0.20259) was voided when the first transfer test showed it could not hold. A transfer test was also voided when it turned out the validation set was one my final competition model had trained on (its velocity error there was only 0.087, against 0.32 to 2.38 on the test data), so that round counted for nothing.

## 5. Process and disclosure

- **Pass/fail rule first.** Twelve files were written before the corresponding submission or check (transfer, stage 2, A, B, C, D, F, G, ground replication, ground members, test-time averaging, uncertainty smoother; the file names use the internal letters). They are archived by me without a third-party timestamp, and for A, D, F and G the rule file and the result file share a code snapshot (commit), so "before" rests on the files' own text.
- **Held-out recordings.** For the ground analysis, 39 recordings (car 11, dog 9, human 7, drone 12) were excluded from both helper modules and the main network. The self-supervised pre-training of the trunk did see their IMU, so this is not strictly clean.
- **Test feedback.** At least six late submissions (A, C, D, E, Step 2, Step 3; the full list was never tabulated) were chosen using the scores Kaggle returned. About 89 per-sequence probe submissions were also made, each altering one test sequence, to recover which sequences are public and which private (41 / 48); the returned score difference of a zeroed sequence equals that sequence's label statistic. The test values measured this way were not used to adjust any output, but the probe results and the 33 official queries did inform diagnosis and some rule boundaries. The organizers have not been asked whether the probe method may be described publicly; I owe them a letter.
- **Selection bits.** D vs E (best of two kept), and C's two extra sequences.
- **Licences.** The borrowed code is Apache-2.0; the readout weights are CC-BY-NC-4.0, so they are not redistributed here.
- **Roles.** Decisions on direction, trade-offs, validation rules and each submission were mine; the written pass/fail rules, runs and recomputations were done by AI coding agents (Claude Code, Codex), with a second agent recomputing the first's results. The proposer of each step is recorded in those rule files only where it was written down.
- **Next step.** I have tried most of the directions I can think of; the next step is to deploy the model on a physical robot and make it a system that works in practice.
