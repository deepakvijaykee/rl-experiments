# Top-k stability

The OPD literature reaches for top-k truncation as an efficiency and stability lever. The question this run asks is what happens to top-k when the overlap assumption it implicitly relies on is violated. A cold-start setting answers that cleanly. With a random student, no SFT alignment, no same-family teacher, and a smoothed oracle teacher, the student has no structural reason yet to share local support with the teacher. That is where the support-restriction trick is most plausibly broken, and where the failure mode should appear in its starkest form.

The top-k variant implemented here is intentionally literal:

```text
sum_{a in student_top_k} pi_student(a|s)
  * (log pi_student(a|s) - log pi_teacher(a|s))
```

It is not renormalized over the selected support. With `k` equal to the full vocabulary, it reduces to ordinary reverse KL. With `k` small, it tests whether the student's current high-probability support already contains the teacher's useful tokens. The non-renormalization is a deliberate diagnostic choice. Renormalizing over the retained support would hide the failure the experiment is trying to expose by making the omitted teacher mass invisible to the loss.

## Command

Run from the repository root:

```bash
python -m opd_sandbox.experiments.topk_stability \
  --top_ks 1,2,4 \
  --num_steps 300 \
  --eval_every 20 \
  --batch_size 64 \
  --num_seeds 3 \
  --vocab_size 8 \
  --seq_len 8 \
  --overlap_k 4 \
  --output_dir opd_sandbox/analysis/results
```

Outputs:

- `opd_sandbox/analysis/results/topk_stability.csv`
- `opd_sandbox/analysis/results/topk_stability.png`
- per-variant CSVs in the same directory

The evidence run completed in about 74 seconds on the local machine.

## Result

Final greedy evaluation at step 300:

| Variant | Test error | Entropy |
| --- | ---: | ---: |
| `full_vocab_rkl` | 0.0045 +/- 0.0069 | 0.0868 +/- 0.0494 |
| `sampled_pg` | 0.6350 +/- 0.0057 | 1.6018 +/- 0.0297 |
| `topk_rkl_k1` | 0.8619 +/- 0.0065 | 2.1971 +/- 0.0000 |
| `topk_rkl_k2` | 0.8538 +/- 0.0051 | 2.1972 +/- 0.0000 |
| `topk_rkl_k4` | 0.8491 +/- 0.0367 | 2.1972 +/- 0.0000 |

Last logged OPD diagnostics at step 280:

| Variant | Reverse KL | Top-1 agreement | Overlap@4 | Reward |
| --- | ---: | ---: | ---: | ---: |
| `full_vocab_rkl` | 0.7436 +/- 0.6037 | 0.9212 +/- 0.0684 | 0.5301 +/- 0.0270 | 0.8939 +/- 0.0835 |
| `sampled_pg` | 4.9129 +/- 0.0184 | 0.3613 +/- 0.0170 | 0.4899 +/- 0.0398 | 0.2578 +/- 0.0167 |
| `topk_rkl_k1` | 5.7917 +/- 0.0010 | 0.1419 +/- 0.0285 | 0.3913 +/- 0.0020 | 0.1087 +/- 0.0141 |
| `topk_rkl_k2` | 5.7913 +/- 0.0007 | 0.1276 +/- 0.0439 | 0.4365 +/- 0.1227 | 0.1087 +/- 0.0141 |
| `topk_rkl_k4` | 5.7902 +/- 0.0005 | 0.1517 +/- 0.0130 | 0.5145 +/- 0.0384 | 0.1074 +/- 0.0137 |

`Overlap@4` is useful only as a rough support diagnostic in this toy. The oracle teacher has one high-probability target token and a uniform tail over wrong tokens, which makes the teacher's non-target top-k entries depend partly on tie-breaking. Top-1 agreement, sampled reward, and teacher mass on the student's selected support carry the cleaner signal.

For the top-k variants, the selected support at step 280 has:

| Variant | Student top-k mass | Teacher mass on student top-k |
| --- | ---: | ---: |
| `topk_rkl_k1` | 0.1130 +/- 0.0003 | 0.1419 +/- 0.0284 |
| `topk_rkl_k2` | 0.2242 +/- 0.0002 | 0.2552 +/- 0.0517 |
| `topk_rkl_k4` | 0.4469 +/- 0.0002 | 0.5299 +/- 0.0417 |

The entropy value `2.197` is approximately `log(9)`, the uniform entropy over the task's nine action tokens. The top-k runs therefore do not collapse to a wrong sharp mode. They fail by staying nearly uniform and never acquiring the teacher's target-token support in the first place.

## Interpretation

Full-vocabulary reverse KL learns the task almost perfectly. Sampled-token OPD improves but stays much slower and noisier. Student-top-k truncation fails from a cold start for every `k` tested, with entropy pinned at the uniform value throughout training. The failure mode is specific enough to point at a mechanism.

The mechanism is in the structure of the reverse-KL gradient itself. For each action in the support, the per-action gradient term is `pi_student(a|s) * (grad log pi_student(a|s) - grad log pi_teacher(a|s))`, and the sum runs across the actions inside the truncated support. Every term is weighted by the student's own probability on that action. Any teacher mass that lies outside the student's top-k contributes exactly zero to the gradient. Early in training the student is essentially uniform, its top-k for any given prefix is largely arbitrary, and the teacher's preferred token frequently falls outside that top-k. Increasing `k` from 1 to 4 raises teacher mass on the selected support from 0.14 to 0.53, but the increase does not enter the regime where the objective acquires the right local asymmetry. The missing component is directional, not quantitative. The optimizer has no signal at all to pull probability toward a token it is not currently considering.

The part most likely to be misread is what counts as "improvement" once the support is restricted. The truncated objective can be reduced toward zero in the sense of its KL value, simply by making the student more uniform on the retained support, while still failing to create the local teacher-student agreement the task requires. The objective and the task disagree on what improvement means once support truncation is in place, and that disagreement is invisible to any single number that only measures the loss.

This is the small-scale analog of the OPD overlap caveat. Top-k OPD is a stability tool only when the student and teacher already share enough local support. In the large-model OPD setting that overlap usually comes for free, from a same-family teacher, an SFT cold start, or both. From an unaligned cold start, top-k truncation can remove the very signal that would have created overlap.

## Scope

What carries over from this toy is the qualitative claim, not the numbers: student-top-k truncation is unreliable as a cold-start objective whenever teacher mass is not already concentrated in the student's selected support. The cold-start setup is what makes the run a stress test of the support-overlap assumption rather than a comparison of efficiency trade-offs. The aligned same-family LLM setups the OPD literature usually describes start from a different regime, where the overlap is given and the question moves on to how to use the budget more efficiently.
