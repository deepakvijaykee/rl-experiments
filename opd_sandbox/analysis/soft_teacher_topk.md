# Soft-teacher top-k support

The earlier top-k runs used a smoothed one-hot teacher. That made the support
failure easy to see, but it also made `Overlap@4` tie-sensitive because every
wrong token had the same probability. This run asks whether the support
truncation finding survives a less artificial teacher:

- The correct token remains the teacher mode.
- Nearby content tokens get graded probability mass.
- The separator/special token gets only a small background mass.

A softer teacher is not automatically a better teacher for this exact-token
metric. It trades a sharper correctness signal for broader support. That
tradeoff is what this experiment exposes.

The comparison this run sets up is between three support choices for the top-k truncation, beyond the simpler full-vocabulary versus top-k contrast:

- `student`: use the student's current top-k tokens.
- `teacher`: use the teacher's current top-k tokens.
- `intersection`: use only tokens present in both top-k sets.

The top-k loss is still an unnormalized reverse-KL contribution over the chosen
support, not a normalized KL over that support.

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

The sharper-teacher sanity check used the same protocol with
`--teacher_temperature 0.5` and output directory
`opd_sandbox/analysis/results/soft_teacher_topk_temp05`.

Outputs:

- `opd_sandbox/analysis/results/soft_teacher_topk.csv`
- `opd_sandbox/analysis/results/soft_teacher_topk.png`
- `opd_sandbox/analysis/results/soft_teacher_topk_temp05/soft_teacher_topk.csv`
- `opd_sandbox/analysis/results/soft_teacher_topk_temp05/soft_teacher_topk.png`

## Final result: broad soft teacher

With `teacher_temperature=1.0`, even full-vocabulary reverse KL does not solve
the exact-token reversal task in 300 steps. The baseline itself is informative:
an argmax-correct teacher can still be too diffuse for the exact reward metric
at a fixed compute budget.

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

The broad teacher makes the support choice look pessimistic across the board.
Student-top-k is least damaging, but it still sits far behind full-vocabulary RKL.
Teacher-top-k and intersection-top-k are especially poor because reverse KL
weights the gradient by the student's probability mass. A token can be in the
teacher top-k and still produce a weak update if the student currently assigns
it little probability.

## Final result: sharper soft teacher

With `teacher_temperature=0.5`, full-vocabulary RKL improves substantially and
the support-choice story becomes more informative:

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

The sharper-teacher run changes the reading. Teacher-support and
intersection-support are better than student-support from a cold start because
they avoid spending all early updates on arbitrary student-preferred tokens. But
after enough full-vocabulary warmup, student-support becomes competitive again:
at 250 warmup steps, student-top-k has the lowest top-k final error among the
truncated variants.

None of the `k=4` variants catches full-vocabulary RKL in this setting. The
support restriction is still too severe for this task and horizon.

## Switch diagnostics

The switch rows make clear why high teacher mass on the selected support is not
enough.

For the broad teacher at 250 warmup steps:

| Support | Switch error | Top-1 agreement | Reward | Student mass | Teacher mass | Support size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| student | 0.6075 | 0.4004 | 0.1738 | 0.6943 | 0.6782 | 4.0000 |
| teacher | 0.6075 | 0.4004 | 0.1738 | 0.5981 | 0.8727 | 4.0000 |
| intersection | 0.6075 | 0.4004 | 0.1738 | 0.4870 | 0.6302 | 2.6536 |

Teacher-support captures much more teacher probability mass, but it produces a
worse final model. Reverse KL is an expectation under the student; teacher mass
outside the student's own high-probability region is not a strong pull by
itself.

For the sharper teacher at cold start:

| Support | Switch error | Top-1 agreement | Reward | Student mass | Teacher mass | Support size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| student | 0.8773 | 0.1009 | 0.1055 | 0.6653 | 0.4604 | 4.0000 |
| teacher | 0.8773 | 0.1009 | 0.1055 | 0.4357 | 0.9685 | 4.0000 |
| intersection | 0.8773 | 0.1009 | 0.1055 | 0.2778 | 0.4459 | 1.6888 |

Here, teacher and intersection support beat student support despite lower
student mass. The teacher is sharp enough that even the restricted off-student
signal is meaningful, while student-support mostly reinforces arbitrary early
student modes.

## Interpretation

This experiment adds three caveats to the earlier hard-oracle top-k story:

1. Teacher entropy matters. A broad teacher can be argmax-correct while still
   being a weak exact-reward teacher at fixed compute. This is not a top-k
   artifact; the full-vocabulary baseline shows it.
2. Support choice depends on alignment. Teacher/intersection support can be
   better at cold start when the teacher is sharp, because student-support
   over-trusts arbitrary student modes. After warmup, student-support can become
   better because reverse KL has usable mass where the student already assigns
   probability.
3. Teacher mass on selected support is not sufficient. Reverse KL is
   student-weighted, so a teacher-top-k set with high teacher mass can still be
   a weak training signal if the student has low mass there.

The transferable claim is therefore narrower and stronger than "top-k needs
warmup":

> Support-truncated OPD needs both a sufficiently sharp teacher signal and a
> sufficiently aligned student support. Which top-k support is least misleading
> depends on which of those two preconditions is missing.

There is no global ordering over support choices in these runs. Student,
teacher, and intersection support each fail for different reasons when the
teacher entropy, student mass, or behavioral alignment precondition is missing.

This is closer to the large-model OPD story than the hard-oracle result alone.
Same-family/cold-start recipes work because the teacher is sharp and locally
meaningful on the student's support, beyond the simpler claim that overlap is
high.

## Scope

These runs do not name a globally preferable top-k support for LLM OPD. They
show why a single overlap or mass-on-support number cannot pick one: support
choice is an interaction between teacher entropy, student mass on that
support, and behavioral alignment. Naming a winner requires measuring all
three together.
