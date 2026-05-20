# Top-k stability

This appendix run is a support-overlap stress test. It compares exact
full-vocabulary reverse KL, sampled-token OPD, and student-top-k truncated
reverse KL from the same cold-start toy transformer.

The question I want answered: what happens to support truncation when the
student has no structural reason yet to share support with the teacher? That
regime is where the overlap assumption most plausibly fails, and where the
support-restriction trick most plausibly breaks. The cold start (random
student, no SFT alignment, no same-family teacher, smoothed oracle
distribution) is the setup that puts that question to the experiment most
directly.

The top-k variant implemented here is intentionally literal:

```text
sum_{a in student_top_k} pi_student(a|s)
  * (log pi_student(a|s) - log pi_teacher(a|s))
```

It is not renormalized over the selected support. When `k` covers the full
vocabulary, it becomes ordinary reverse KL. When `k` is small, it tests whether
the student's current high-probability support already contains the teacher's
useful tokens.

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

`Overlap@4` is useful only as a rough support diagnostic in this toy. The
oracle teacher has one high-probability target token and a uniform tail over
wrong tokens, so the teacher's non-target top-k entries are partly determined
by tie-breaking. Top-1 agreement, reward, and teacher mass on the student's
selected support are more meaningful here.

For the top-k variants, the selected support at step 280 has:

| Variant | Student top-k mass | Teacher mass on student top-k |
| --- | ---: | ---: |
| `topk_rkl_k1` | 0.1130 +/- 0.0003 | 0.1419 +/- 0.0284 |
| `topk_rkl_k2` | 0.2242 +/- 0.0002 | 0.2552 +/- 0.0517 |
| `topk_rkl_k4` | 0.4469 +/- 0.0002 | 0.5299 +/- 0.0417 |

The entropy value `2.197` is approximately `log(9)`, the uniform entropy over
the task's nine action tokens. The top-k runs therefore do not collapse to a
wrong sharp mode; they fail by staying almost uniform and not acquiring the
teacher's target-token support.

## Interpretation

This run is informative because the failure mode is specific. Full-vocabulary
reverse KL learns the task almost perfectly. Sampled-token OPD improves but
remains much slower and noisier. The student-top-k truncated objective fails
from a cold start for all tested `k`.

The mechanism is support mismatch. Early in training, the student's top-k set
often omits the teacher's correct token. Once the objective only looks at the
student-selected support, it cannot reliably pull probability toward omitted
teacher-preferred tokens. Increasing `k` from 1 to 4 increases teacher mass on
the selected support, but not enough to enter the successful regime.

What is missing here is directional asymmetry, not mode-collapsed probability mass. The objective can become easy to reduce while still failing to create the local teacher-student agreement that the task requires.

This is the small-scale analog of the OPD overlap caveat: top-k OPD is a
stability tool only when the student and teacher already share enough local
support. In the successful large-model setting, that overlap usually comes
from a same-family teacher, an SFT cold start, or both. From an unaligned cold
start, top-k truncation can remove the very signal needed to create overlap.

## Scope

The transferable claim is that student-top-k truncation becomes unreliable as
a cold-start objective when teacher mass is not already concentrated in the
student's selected support. The cold-start setup is what makes the run a
stress test of the support-overlap assumption. The aligned same-family LLM
setups the OPD literature usually describes start from a different regime,
where the overlap is given and the question moves on to efficiency.
