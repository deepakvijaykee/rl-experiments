# Top-K Warmup Sweep

This appendix run asks whether there is a measurable switch threshold for
top-k OPD. The previous cold-start run showed that switching after 100
full-vocabulary steps was too early. Here we sweep the warmup length and test
two truncation widths:

- `k=4`, a genuinely narrow support restriction over 9 action tokens.
- `k=8`, almost full support over 9 action tokens.

Those two values ask different questions. `k=8` excludes only one action token,
so it is a mild regularizer. `k=4` is a real bottleneck. Treating both as
"top-k" without this distinction would hide the mechanism.

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

## Final Result

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

The threshold is visible but depends strongly on support width. With `k=8`,
top-k is nearly full-vocabulary and becomes close to stable after 150 warmup
steps, then matches the full-vocabulary baseline by 200 steps. With `k=4`,
the method only becomes stable after 250 warmup steps.

## Switch Diagnostics

At the switch row, before the first top-k update:

| Warmup steps | Switch error | Top-1 agreement | Reward | Reverse KL |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8773 | 0.1009 | 0.1055 | 5.9596 |
| 50 | 0.7495 | 0.2402 | 0.1465 | 5.5967 |
| 100 | 0.6160 | 0.3691 | 0.2949 | 4.8203 |
| 150 | 0.5830 | 0.4160 | 0.3294 | 4.5964 |
| 200 | 0.5144 | 0.4824 | 0.4069 | 4.1331 |
| 250 | 0.2620 | 0.7318 | 0.6842 | 2.2702 |

The switch diagnostics are the same for `k=4` and `k=8` because the warmup
phase is full-vocabulary in both cases.

As in the earlier top-k notes, `Overlap@4` is tie-sensitive in this oracle
teacher because every wrong token has the same probability. It helps expose
gross support mismatch, but it should not be used as the sole switch rule.

For the actual selected support at the switch:

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

Mass-on-support alone is not a sufficient switch criterion. At `k=4`,
teacher mass on the student's selected support is already about 0.90 after 200
warmup steps, but final error is still 0.61. The 250-step switch is different:
it has both high support mass and much better behavioral alignment, with top-1
agreement 0.73 and sampled reward 0.68.

## Post-Switch Change

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

For `k=8`, switching usually continues improving the model because the support
restriction is weak: it excludes only one action token. For `k=4`, switching
between 50 and 200 steps degrades the model, confirming that partial warmup can
be erased by too-early truncation.

## Interpretation

This sweep sharpens the previous two top-k results:

> Top-k OPD becomes viable only after the student enters a sufficiently aligned
> support regime, and the required alignment threshold rises as k gets smaller.

In this toy setting, `k=8` is almost full support and becomes stable well before
the model has solved the task. `k=4` needs a much stronger warmup. The most
useful switch diagnostics are behavioral: top-1 agreement and sampled reward.
Overlap@4 and mass-on-support help describe the geometry, but they do not by
themselves predict stable switching.

This is the small-scale analog of the same-family/cold-start lesson in OPD:
top-k truncation is an efficiency or stability device after the student and
teacher have entered a shared local support. It is not a reliable way to create
that support from scratch.

The numeric threshold is not universal. It is a property of this teacher, this
student, this horizon, and this compute budget. The transferable part is the
diagnostic structure: support mass, entropy, and behavior need to move together.
High retained mass alone is not proof that the truncated objective still
contains the useful correction.

## Scope

This is still a toy oracle-teacher result. It does not reproduce top-k OPD in
large LLMs and does not imply that the numerical thresholds transfer. The
transferable claim is mechanistic: support truncation has a precondition, and
the precondition is measurable.
