# Top-k warmup sweep

The cold-start probe established that 100 full-vocabulary steps is too short a warmup to stabilize student-top-k truncation. The natural next move is to sweep warmup length more thoroughly and find the threshold. Doing the sweep at a single truncation width would only tell part of the story, because the meaning of "top-k" depends on how aggressive the truncation is. This run tests two widths in parallel:

- `k=4`, which covers less than half the action space.
- `k=8`, which covers all but one action token.

`k=8` is a mild regularizer over the nine-token vocabulary; it excludes only one action, and rarely the teacher's preferred one once any warmup has happened. `k=4` cuts the policy down to roughly half its action space, which is a different kind of intervention. Averaging both under the label "top-k" would hide the mechanism, because the threshold at which either becomes stable depends on which one is being used.

## Command

Run from the repository root:

```bash
python -m opd_sandbox.experiments.topk_warmup_sweep \
  --top_ks 4,8 \
  --warmup_steps 0,50,100,150,200,250 \
  --num_steps 300 \
  --eval_every 10 \
  --batch_size 64 \
  --num_seeds 3 \
  --vocab_size 8 \
  --seq_len 8 \
  --overlap_k 4 \
  --output_dir opd_sandbox/analysis/results
```

Outputs:

- `opd_sandbox/analysis/results/topk_warmup_sweep.csv`
- `opd_sandbox/analysis/results/topk_warmup_sweep.png`
- per-variant CSVs in the same directory

The evidence run completed in about 238 seconds on the local machine.

## Final result

Final greedy test error at step 300:

| Top-k | Warmup steps | Final test error | Final entropy |
| ---: | ---: | ---: | ---: |
| full vocab | all | 0.0022 +/- 0.0020 | 0.0682 +/- 0.0298 |
| 4 | 0 | 0.8484 +/- 0.0142 | 2.1972 +/- 0.0000 |
| 4 | 50 | 0.7760 +/- 0.0122 | 2.1972 +/- 0.0000 |
| 4 | 100 | 0.7068 +/- 0.0435 | 1.9846 +/- 0.0153 |
| 4 | 150 | 0.7148 +/- 0.0318 | 2.0090 +/- 0.0510 |
| 4 | 200 | 0.6131 +/- 0.0688 | 1.9049 +/- 0.0580 |
| 4 | 250 | 0.0079 +/- 0.0064 | 0.1808 +/- 0.1172 |
| 8 | 0 | 0.3322 +/- 0.0259 | 0.9376 +/- 0.0476 |
| 8 | 50 | 0.3029 +/- 0.0492 | 0.8470 +/- 0.1497 |
| 8 | 100 | 0.1643 +/- 0.1480 | 0.5225 +/- 0.3742 |
| 8 | 150 | 0.0279 +/- 0.0369 | 0.1882 +/- 0.1468 |
| 8 | 200 | 0.0022 +/- 0.0023 | 0.0779 +/- 0.0342 |
| 8 | 250 | 0.0011 +/- 0.0010 | 0.0692 +/- 0.0327 |

A threshold is visible, and it depends sharply on the truncation width. With `k=8`, the truncation is close to full-vocabulary and becomes nearly stable after 150 warmup steps, matching the full-vocabulary baseline by 200. With `k=4`, the same method only becomes stable after 250 warmup steps. The interesting question is what the difference between those two thresholds reflects about the underlying mechanism.

## Switch diagnostics

At the switch row, before the first top-k update:

| Warmup steps | Switch error | Top-1 agreement | Reward | Reverse KL |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8773 | 0.1009 | 0.1055 | 5.9596 |
| 50 | 0.7495 | 0.2402 | 0.1465 | 5.5967 |
| 100 | 0.6160 | 0.3691 | 0.2949 | 4.8203 |
| 150 | 0.5830 | 0.4160 | 0.3294 | 4.5964 |
| 200 | 0.5144 | 0.4824 | 0.4069 | 4.1331 |
| 250 | 0.2620 | 0.7318 | 0.6842 | 2.2702 |

The switch diagnostics are identical for `k=4` and `k=8` because the warmup phase is full-vocabulary in both cases. As in the earlier top-k notes, `Overlap@4` is tie-sensitive in this oracle teacher because every wrong token has the same probability, so it can flag gross support mismatch but should not be used as the sole switch rule.

For the selected support at the switch:

| Top-k | Warmup steps | Student top-k mass | Teacher mass on student top-k |
| ---: | ---: | ---: | ---: |
| 4 | 0 | 0.6653 | 0.4622 |
| 4 | 50 | 0.6024 | 0.6976 |
| 4 | 100 | 0.7869 | 0.7906 |
| 4 | 150 | 0.8597 | 0.8550 |
| 4 | 200 | 0.8914 | 0.8966 |
| 4 | 250 | 0.9626 | 0.9636 |
| 8 | 0 | 0.9595 | 0.8893 |
| 8 | 50 | 0.9641 | 0.9999 |
| 8 | 100 | 0.9764 | 0.9986 |
| 8 | 150 | 0.9839 | 0.9953 |
| 8 | 200 | 0.9879 | 0.9979 |
| 8 | 250 | 0.9972 | 0.9992 |

Mass-on-support alone is not a sufficient switch criterion. At `k=4`, teacher mass on the student's selected support is already 0.90 by 200 warmup steps, but final error is still 0.61. The 250-step switch is qualitatively different in a way mass-on-support does not capture: it pairs high support mass with much stronger behavioral alignment, with top-1 agreement 0.73 and sampled reward 0.68, and that combination is what lets the truncated objective converge.

## Post-switch change

Final error minus switch error:

| Top-k | Warmup steps | Error change after switch |
| ---: | ---: | ---: |
| 4 | 0 | -0.0289 |
| 4 | 50 | +0.0265 |
| 4 | 100 | +0.0907 |
| 4 | 150 | +0.1318 |
| 4 | 200 | +0.0988 |
| 4 | 250 | -0.2541 |
| 8 | 0 | -0.5451 |
| 8 | 50 | -0.4466 |
| 8 | 100 | -0.4518 |
| 8 | 150 | -0.5551 |
| 8 | 200 | -0.5122 |
| 8 | 250 | -0.2609 |

For `k=8`, switching continues to improve the model at every warmup length, because the support restriction is mild enough that excluding one action token is rarely consequential once any warmup has happened. For `k=4`, switching between 50 and 200 steps actively degrades the model, which is the partial-warmup-erasure pattern from the previous run, made visible across the full warmup sweep.

## Interpretation

The reading I take from this sweep is that top-k OPD becomes viable only after the student enters a sufficiently aligned support regime, and the required alignment threshold rises as `k` gets smaller. With `k=8` the support restriction is mild, and the threshold sits below where the model has even solved the task. With `k=4` the support restriction is severe, and the threshold rises to the point where even high teacher mass on the retained support is not enough to guarantee a useful gradient.

The most useful switch diagnostics are the behavioral ones, meaning top-1 agreement and sampled reward. Overlap@4 and mass-on-support describe the geometry but they do not on their own predict stable switching. The reason is mechanical. High teacher mass on the student's selected support can coexist with the student spreading its own mass nearly uniformly across that support, and the per-action gradient on each term scales with `π_student(a)`. A flat student distribution produces small per-term updates even when the geometry looks healthy. Top-1 agreement measures the condition that actually concentrates `π_student` on the teacher's preferred token, and that concentration is what lets the truncated objective converge on the teacher rather than dilute around it.

This is the small-scale analog of the same-family / cold-start lesson in OPD. Top-k truncation belongs in the efficiency-and-stability toolbox once the student and teacher share a local support; it does not create that support from scratch.

The numeric threshold here is not universal. It is a property of this teacher, this student, this horizon, and this compute budget. The transferable part is the diagnostic structure: support mass, entropy, and behavior have to move together, and high retained mass alone is not evidence that the truncated objective still contains the useful correction.

## Scope

The numeric thresholds in this sweep belong to this teacher, student, horizon, and compute budget. The mechanistic claim travels more broadly: support truncation has a precondition, the precondition is measurable on the same diagnostics the run uses, and the threshold tightens as the retained support gets narrower.
