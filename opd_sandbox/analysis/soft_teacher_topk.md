# Soft-teacher top-k support

The earlier top-k runs all used a smoothed one-hot oracle teacher. That choice made the support-overlap failure mode easy to see, because the teacher had a single correct token and a uniform tail over the wrong ones. The cost was that the toy was artificially neat. `Overlap@4` became tie-sensitive, since every wrong token shared the same probability, and the support choice was forced to live or die on whether the one correct token landed inside the retained top-k.

This run asks whether the support-truncation finding survives a less artificial teacher whose support carries graded structure across more than one token. The graded teacher does the following:

- The correct token remains the teacher mode.
- Nearby content tokens carry graded probability mass.
- The separator/special token carries only a small background mass.

A softer teacher is not automatically a better teacher for the exact-token reward used here. It trades a sharper correctness signal for broader support, and the gradient pulled by reverse KL has both a magnitude and a direction component. Broadening the teacher attenuates the directional component at every position even when its argmax is correct, which is a property worth holding in mind when reading the numbers below.

The comparison extends the earlier two-way contrast (full vocabulary versus top-k) into a three-way contrast over how the top-k support itself is chosen:

- `student`: the student's current top-k tokens.
- `teacher`: the teacher's current top-k tokens.
- `intersection`: only tokens present in both top-k sets.

The top-k loss remains the unnormalized reverse-KL contribution over the chosen support, keeping the diagnostic structure consistent with the earlier runs.

## Commands

Run from the repository root:

```bash
python -m opd_sandbox.experiments.soft_teacher_topk \
  --top_ks 4 \
  --supports student,teacher,intersection \
  --warmup_steps 0,100,200,250 \
  --num_steps 300 \
  --eval_every 10 \
  --batch_size 64 \
  --num_seeds 3 \
  --vocab_size 8 \
  --seq_len 8 \
  --teacher_temperature 1.0 \
  --special_token_weight 0.02 \
  --overlap_k 4 \
  --output_dir opd_sandbox/analysis/results
```

The sharper-teacher sanity check used the same protocol with `--teacher_temperature 0.5` and output directory `opd_sandbox/analysis/results/soft_teacher_topk_temp05`.

The two runs write `opd_sandbox/analysis/results/soft_teacher_topk.csv` and `opd_sandbox/analysis/results/soft_teacher_topk.png` for the broad teacher, and `opd_sandbox/analysis/results/soft_teacher_topk_temp05/soft_teacher_topk.csv` with `opd_sandbox/analysis/results/soft_teacher_topk_temp05/soft_teacher_topk.png` for the sharper one, so the two temperatures produce parallel result sets that are read against each other below.

## Final result: broad soft teacher

With `teacher_temperature=1.0`, even full-vocabulary reverse KL does not solve the exact-token reversal task in 300 steps. The baseline itself carries information: an argmax-correct teacher can be too diffuse for the exact reward metric at fixed compute. Any reading of the support-choice rows below has to land against that baseline.

| Support | Warmup steps | Final test error | Final entropy |
| --- | ---: | ---: | ---: |
| full vocab | all | 0.5112 +/- 0.0993 | 1.9169 +/- 0.0676 |
| student | 0 | 0.8351 +/- 0.0151 | 2.1972 +/- 0.0000 |
| student | 100 | 0.7288 +/- 0.0245 | 2.1972 +/- 0.0000 |
| student | 200 | 0.6945 +/- 0.0058 | 2.1968 +/- 0.0000 |
| student | 250 | 0.7160 +/- 0.0100 | 2.1945 +/- 0.0008 |
| teacher | 0 | 1.0000 +/- 0.0000 | 1.8517 +/- 0.0075 |
| teacher | 100 | 0.9999 +/- 0.0002 | 1.8601 +/- 0.0114 |
| teacher | 200 | 1.0000 +/- 0.0000 | 1.8572 +/- 0.0282 |
| teacher | 250 | 0.9173 +/- 0.0639 | 2.0375 +/- 0.0306 |
| intersection | 0 | 1.0000 +/- 0.0000 | 1.7644 +/- 0.0462 |
| intersection | 100 | 1.0000 +/- 0.0000 | 2.0131 +/- 0.0192 |
| intersection | 200 | 0.8585 +/- 0.0135 | 2.1413 +/- 0.0410 |
| intersection | 250 | 0.7424 +/- 0.0143 | 2.1577 +/- 0.0195 |

Reading down the support blocks rather than across the warmup columns is what exposes the ordering, since within each support the warmup trend is mild compared to the gap between supports.

The broad teacher makes every support choice look pessimistic. Student-top-k is the least damaging of the truncated variants but still trails full-vocabulary RKL by a wide margin. Teacher-top-k and intersection-top-k are worse, and the reason is the student-weighting in the reverse-KL gradient. A token can sit inside the teacher's top-k and still produce a weak update if the student currently assigns it little probability. Concentrating the retained support on the teacher's side of the picture does not help when the bottleneck is on the student's side.

## Final result: sharper soft teacher

With `teacher_temperature=0.5`, full-vocabulary RKL improves substantially and the support-choice story becomes more informative:

| Support | Warmup steps | Final test error | Final entropy |
| --- | ---: | ---: | ---: |
| full vocab | all | 0.2658 +/- 0.0476 | 1.3172 +/- 0.0428 |
| student | 0 | 0.8471 +/- 0.0236 | 2.1972 +/- 0.0000 |
| student | 100 | 0.7292 +/- 0.0079 | 2.1971 +/- 0.0000 |
| student | 200 | 0.7085 +/- 0.0329 | 2.1666 +/- 0.0301 |
| student | 250 | 0.6773 +/- 0.0204 | 2.1519 +/- 0.0133 |
| teacher | 0 | 0.6768 +/- 0.0022 | 1.9047 +/- 0.0052 |
| teacher | 100 | 0.6836 +/- 0.0098 | 1.9455 +/- 0.0061 |
| teacher | 200 | 0.6651 +/- 0.0337 | 1.9887 +/- 0.0102 |
| teacher | 250 | 0.7358 +/- 0.0270 | 2.0624 +/- 0.0168 |
| intersection | 0 | 0.6631 +/- 0.0109 | 2.1002 +/- 0.0091 |
| intersection | 100 | 0.6358 +/- 0.0043 | 2.1744 +/- 0.0009 |
| intersection | 200 | 0.6394 +/- 0.0380 | 2.1768 +/- 0.0104 |
| intersection | 250 | 0.7131 +/- 0.0435 | 2.1792 +/- 0.0106 |

The ordering inverts relative to the broad-teacher table, which is the result worth pausing on, because nothing about the support-selection rule changed between the two runs.

The sharper-teacher run changes the reading. Teacher-support and intersection-support are better than student-support at cold start, because student-support spends its early updates reinforcing arbitrary student-preferred tokens that have no relationship to the teacher's preference. After enough full-vocabulary warmup the picture flips. Student-support becomes competitive again, and at 250 warmup steps it has the lowest top-k final error among the truncated variants. None of the `k=4` variants catches full-vocabulary RKL, because the support restriction is still too severe for the task and horizon.

## Switch diagnostics

The switch rows expose why high teacher mass on the selected support is not enough on its own to predict stable switching.

Taking the broad teacher at 250 warmup steps first, every row below shares an identical switch state, since the warmup phase is full-vocabulary regardless of which support the truncation will later use. Only the last three columns differ, which makes them the only candidates for predicting the outcome.

| Support | Switch error | Top-1 agreement | Reward | Student mass | Teacher mass | Support size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| student | 0.6075 | 0.4004 | 0.1738 | 0.6943 | 0.6782 | 4.0000 |
| teacher | 0.6075 | 0.4004 | 0.1738 | 0.5981 | 0.8727 | 4.0000 |
| intersection | 0.6075 | 0.4004 | 0.1738 | 0.4870 | 0.6302 | 2.6536 |

Teacher-support captures the most teacher probability mass and produces the worst final model. Reverse KL is an expectation under the student, and teacher mass outside the student's own high-probability region does not pull strongly enough to recover.

The sharper teacher at cold start puts the same three columns under the opposite verdict.

| Support | Switch error | Top-1 agreement | Reward | Student mass | Teacher mass | Support size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| student | 0.8773 | 0.1009 | 0.1055 | 0.6653 | 0.4604 | 4.0000 |
| teacher | 0.8773 | 0.1009 | 0.1055 | 0.4357 | 0.9685 | 4.0000 |
| intersection | 0.8773 | 0.1009 | 0.1055 | 0.2778 | 0.4459 | 1.6888 |

Here teacher and intersection support beat student support despite lower student mass on the retained set. The teacher is sharp enough that even the restricted off-student signal is meaningful, while student-support mostly reinforces arbitrary early student modes.

## Interpretation

The hard-oracle top-k story acquires three caveats once the teacher is allowed to be soft. The first is that teacher entropy is a first-order variable. A broad teacher can be argmax-correct and still be a weak exact-reward teacher at fixed compute, because the reverse-KL gradient carries a directional component proportional to how much `log pi_teacher(a)` varies across the support. A broad teacher attenuates that variation, so the per-token pull shrinks even when the teacher's argmax is right. The full-vocabulary baseline in the broad-teacher table is the cleanest witness for that, and the effect is not a top-k artifact.

The second is that support choice depends on alignment, not on teacher mass alone. Teacher-support and intersection-support can be better than student-support at cold start when the teacher is sharp, because student-support over-trusts arbitrary student modes. After warmup, student-support can become competitive again because reverse KL has usable mass where the student already assigns probability. Neither support choice is globally better.

The third is that teacher mass on the selected support is necessary but not sufficient. A teacher-top-k set with high teacher mass can still produce a weak training signal if the student has low mass there. The trade-off between switching to teacher-support and staying on student-support is the trade between teacher-signal density and student-mass weight.

The takeaway is therefore narrower and stronger than "top-k needs warmup". Support-truncated OPD needs both a sufficiently sharp teacher signal and a sufficiently aligned student support, and which top-k support is least misleading depends on which of the two preconditions is missing. There is no global ordering over support choices in these runs. Student, teacher, and intersection support each fail for different reasons when the teacher entropy, student mass, or behavioral alignment precondition is missing.

This is closer to the large-model OPD story than the hard-oracle result alone. The reason same-family / cold-start OPD recipes work in practice is that the teacher is sharp and the local support is meaningful for the student, beyond the simpler claim that overlap is high.

## Scope

No globally preferable top-k support for LLM OPD comes out of these runs, and the more useful result is the reason why: a single overlap or mass-on-support number does not contain enough information to pick one. The choice is an interaction between teacher entropy, student mass on the retained support, and behavioral alignment, so picking a winner means measuring all three together.
