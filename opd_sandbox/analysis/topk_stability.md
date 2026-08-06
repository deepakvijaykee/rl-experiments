# Top-k stability

The OPD literature reaches for top-k truncation as an efficiency and stability lever. The question this run asks is what happens to top-k when the overlap assumption it implicitly relies on is violated. A cold-start setting answers that cleanly. With a random student, no SFT alignment, no same-family teacher, and a smoothed oracle teacher, the student has no structural reason yet to share local support with the teacher. That is where the support-restriction trick is most plausibly broken, and where the failure mode should appear in its starkest form.

The top-k variant implemented here is intentionally literal:

```math
\sum_{a \in \text{student top-}k} \pi_\text{student}(a \mid s)\,
\bigl(\log \pi_\text{student}(a \mid s) - \log \pi_\text{teacher}(a \mid s)\bigr)
```

It is not renormalized over the selected support. With $k$ equal to the full vocabulary it reduces to ordinary reverse KL, and with $k$ small it tests whether the student's current high-probability support already contains the teacher's useful tokens. Leaving it unnormalized is what keeps the diagnostic honest, since renormalizing over the retained support would make the omitted teacher mass invisible to the loss, and that omitted mass is the whole subject of the experiment.

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

This writes `opd_sandbox/analysis/results/topk_stability.csv`, `opd_sandbox/analysis/results/topk_stability.png`, and per-variant CSVs in the same directory, and it finishes in about 74 seconds locally. That runtime is most of the argument for asking the question here first. Whether support truncation has a precondition is settleable over a coffee break at this scale, and it would cost a GPU-week to settle at any other.

## Result

The headline comparison is the final greedy evaluation at step 300, where the question is simply whether each variant learned the task at all. Every table below reports the mean across three seeds, with the standard error in parentheses.

| Variant | Test error | Entropy |
| --- | ---: | ---: |
| `full_vocab_rkl` | 0.0045 (0.0069) | 0.0868 (0.0494) |
| `sampled_pg` | 0.6350 (0.0057) | 1.6018 (0.0297) |
| `topk_rkl_k1` | 0.8619 (0.0065) | 2.1971 (0.0000) |
| `topk_rkl_k2` | 0.8538 (0.0051) | 2.1972 (0.0000) |
| `topk_rkl_k4` | 0.8491 (0.0367) | 2.1972 (0.0000) |

Those error numbers say the top-k runs failed but not how. The diagnostics logged at step 280 narrow it down, and the column to watch is top-1 agreement, which measures whether the student has come to prefer the same token the teacher does.

| Variant | Reverse KL | Top-1 agreement | Overlap@4 | Reward |
| --- | ---: | ---: | ---: | ---: |
| `full_vocab_rkl` | 0.7436 (0.6037) | 0.9212 (0.0684) | 0.5301 (0.0270) | 0.8939 (0.0835) |
| `sampled_pg` | 4.9129 (0.0184) | 0.3613 (0.0170) | 0.4899 (0.0398) | 0.2578 (0.0167) |
| `topk_rkl_k1` | 5.7917 (0.0010) | 0.1419 (0.0285) | 0.3913 (0.0020) | 0.1087 (0.0141) |
| `topk_rkl_k2` | 5.7913 (0.0007) | 0.1276 (0.0439) | 0.4365 (0.1227) | 0.1087 (0.0141) |
| `topk_rkl_k4` | 5.7902 (0.0005) | 0.1517 (0.0130) | 0.5145 (0.0384) | 0.1074 (0.0137) |

`Overlap@4` is useful only as a rough support diagnostic in this toy. The oracle teacher has one high-probability target token and a uniform tail over wrong tokens, which makes the teacher's non-target top-k entries depend partly on tie-breaking. Top-1 agreement, sampled reward, and teacher mass on the student's selected support carry the cleaner signal.

That leaves one candidate explanation to eliminate. If the truncated runs simply retained too little teacher mass, then widening $k$ should have rescued them, so it is worth looking at how much teacher probability each selected support actually captured at step 280.

| Variant | Student top-k mass | Teacher mass on student top-k |
| --- | ---: | ---: |
| `topk_rkl_k1` | 0.1130 (0.0003) | 0.1419 (0.0284) |
| `topk_rkl_k2` | 0.2242 (0.0002) | 0.2552 (0.0517) |
| `topk_rkl_k4` | 0.4469 (0.0002) | 0.5299 (0.0417) |

The entropy value $2.197$ is approximately $\log 9$, the uniform entropy over the task's nine action tokens. The top-k runs therefore do not collapse to a wrong sharp mode. They fail by staying nearly uniform and never acquiring the teacher's target-token support in the first place.

## Interpretation

Full-vocabulary reverse KL learns the task almost perfectly, and sampled-token OPD gets there too, much more slowly and noisily. Student-top-k truncation does not get there at all: it fails from a cold start for every $k$ tested, with entropy pinned at the uniform value throughout training. A failure that specific points at a mechanism, and the mechanism sits in the structure of the reverse-KL gradient.

For each action in the support, the per-action gradient term is `pi_student(a|s) * (grad log pi_student(a|s) - grad log pi_teacher(a|s))`, summed across the actions inside the truncated support. Every term carries a weight of the student's own probability on that action, so teacher mass lying outside the student's top-k contributes exactly zero. Early in training the student is essentially uniform, its top-k for any given prefix is largely arbitrary, and the teacher's preferred token frequently falls outside it. Widening $k$ from 1 to 4 does raise teacher mass on the selected support from 0.14 to 0.53, and it never reaches the regime where the objective acquires the right local asymmetry, because what is missing is directional and no amount of retained mass supplies a direction. The optimizer gets no signal whatsoever pulling probability toward a token it is not currently considering.

A second and quieter problem sits underneath that one. The truncated objective and the task stop agreeing about what improvement means, because the truncated KL value can be driven toward zero simply by spreading the student more uniformly over the retained support. That satisfies the loss while creating none of the local teacher-student agreement the task requires, so any single number tracking only the loss will report progress straight through the failure.

This is the small-scale analog of the OPD overlap caveat. Top-k OPD is a stability tool only when the student and teacher already share enough local support. In the large-model OPD setting that overlap usually comes for free, from a same-family teacher, an SFT cold start, or both. From an unaligned cold start, top-k truncation can remove the very signal that would have created overlap.

## Scope

What carries over from this toy is the qualitative claim, not the numbers: student-top-k truncation is unreliable as a cold-start objective whenever teacher mass is not already concentrated in the student's selected support. The cold-start setup is what makes the run a stress test of the support-overlap assumption rather than a comparison of efficiency trade-offs. The aligned same-family LLM setups the OPD literature usually describes start from a different regime, where the overlap is given and the question moves on to how to use the budget more efficiently.
