# Top-k cold start

The cold-start stability run left a natural follow-up. If top-k truncation fails from a random start because the student's selected support omits the teacher's useful token, would a short full-vocabulary warmup pre-arrange enough overlap to make top-k stable afterward? The mechanism behind the cold-start failure was that the gradient cannot pull probability toward tokens it does not see. If a warmup phase puts those tokens inside the retained support before the truncation kicks in, the truncation should work from the warmed-up state even though it does not work from the cold-start state. This run tests that proposition with a 100-step warmup probe.

Three arms run side by side:

- `full_vocab_rkl`: full-vocabulary reverse KL for all 300 steps. Reference.
- `cold_topk_k{1,2,4}`: student-top-k truncated reverse KL from step 0. Reproduces the stability result for direct comparison against the warmed-up arm.
- `warm_topk_k{1,2,4}`: full-vocabulary reverse KL for 100 steps, then the matching top-k truncated objective for the remaining 200.

The 100-step warmup is a probe, not a recommended schedule. It is long enough to create visible behavioral alignment and short enough that the switch to top-k can still stress the support-overlap assumption.

## Command

Run from the repository root:

```bash
python -m opd_sandbox.experiments.topk_cold_start \
  --top_ks 1,2,4 \
  --num_steps 300 \
  --warmup_steps 100 \
  --eval_every 20 \
  --batch_size 64 \
  --num_seeds 3 \
  --vocab_size 8 \
  --seq_len 8 \
  --overlap_k 4 \
  --output_dir opd_sandbox/analysis/results
```

This writes `opd_sandbox/analysis/results/topk_cold_start.csv`, `opd_sandbox/analysis/results/topk_cold_start.png`, and per-variant CSVs in the same directory, in about 106 seconds locally.

## Result

The warm and cold arms sit side by side in the final greedy evaluation at step 300, so the comparison to make is whether any warm row has escaped the near-uniform entropy that characterized the cold-start failure. Every table below reports the mean across three seeds, with the standard error in parentheses.

| Variant | Test error | Entropy |
| --- | ---: | ---: |
| `full_vocab_rkl` | 0.0010 (0.0012) | 0.0608 (0.0227) |
| `cold_topk_k1` | 0.8670 (0.0136) | 2.1971 (0.0001) |
| `cold_topk_k2` | 0.8666 (0.0190) | 2.1972 (0.0001) |
| `cold_topk_k4` | 0.8448 (0.0230) | 2.1972 (0.0000) |
| `warm_topk_k1` | 0.7981 (0.0166) | 2.1969 (0.0001) |
| `warm_topk_k2` | 0.7749 (0.0440) | 2.1971 (0.0001) |
| `warm_topk_k4` | 0.6963 (0.0166) | 1.9873 (0.0185) |

The step-280 diagnostics show how far each arm got behaviorally, which matters because the entropy column above cannot distinguish a student that never moved from one that moved and came back.

| Variant | Reverse KL | Top-1 agreement | Overlap@4 | Reward |
| --- | ---: | ---: | ---: | ---: |
| `full_vocab_rkl` | 0.6645 (0.5742) | 0.9355 (0.0609) | 0.5415 (0.0299) | 0.8971 (0.0827) |
| `cold_topk_k1` | 5.7899 (0.0014) | 0.1432 (0.0158) | 0.4508 (0.0456) | 0.1094 (0.0220) |
| `cold_topk_k2` | 5.7906 (0.0006) | 0.1491 (0.0079) | 0.4316 (0.0220) | 0.1107 (0.0209) |
| `cold_topk_k4` | 5.7904 (0.0002) | 0.1549 (0.0041) | 0.4740 (0.0421) | 0.1100 (0.0214) |
| `warm_topk_k1` | 5.7837 (0.0013) | 0.1940 (0.0502) | 0.5099 (0.0698) | 0.1107 (0.0209) |
| `warm_topk_k2` | 5.7832 (0.0025) | 0.2389 (0.0127) | 0.4468 (0.0630) | 0.1113 (0.0220) |
| `warm_topk_k4` | 5.1169 (0.0304) | 0.2747 (0.0285) | 0.5535 (0.0249) | 0.2083 (0.0108) |

`Overlap@4` is tie-sensitive in this oracle task because every wrong teacher token has equal probability. It is useful as a rough geometry check but not as a switch criterion on its own. Top-1 agreement and reward carry the cleaner behavioral signal.

The clearest evidence sits in the trajectory rather than the endpoint. Because all warm arms share an identical full-vocabulary phase, any divergence after step 100 is attributable to the switch itself, and the three steps around it are enough to see what the switch does.

| Variant | Step 100 test error | Step 120 test error | Step 280 test error |
| --- | ---: | ---: | ---: |
| `full_vocab_rkl` | 0.6243 | 0.5978 | 0.0574 |
| `warm_topk_k1` | 0.6243 | 0.7060 | 0.8079 |
| `warm_topk_k2` | 0.6243 | 0.6983 | 0.7575 |
| `warm_topk_k4` | 0.6243 | 0.6306 | 0.7118 |

All warm variants share the same full-vocabulary trajectory through step 100, by construction. After the switch, $k=1$ and $k=2$ degrade almost immediately. $k=4$ degrades more slowly and retains some of the warmup's improvement, but even $k=4$ ends up far from the full-vocabulary baseline.

## Interpretation

A 100-step warmup creates partial overlap, but not enough overlap to make student-top-k truncation stable. At the switch point, full-vocabulary OPD has only reached about 0.37 top-1 agreement and 0.29 sampled reward. That is materially better than the cold start, but it is still far from the regime where the student's top-k reliably contains the teacher's preferred token. Once the objective is restricted to that top-k, the missing teacher-preferred tokens again stop receiving reliable gradient, and the corrective signal the warmup had begun to put in place is now invisible to the loss.

The sharper observation is that switching too early erases a partial warmup. The switch is itself an intervention, and what lets it undo the warmup is the same $\pi_\text{student}$-weighting that drove the cold-start failure. Any teacher mass the warmup has not yet pulled into the student's top-k stops receiving gradient the instant truncation engages, and the optimizer cannot pull it back without seeing it. So when the switch removes the omitted-token signal before the behavior has anchored, the top-k phase actively undoes the asymmetry the warmup had just produced and the model drifts back toward uniformity. The warm $k=1$ and $k=2$ rows show this most cleanly, sitting at the full-vocabulary baseline error of 0.62 at step 100 and degrading from the moment truncation engages.

The mechanism behind both arms is the same, sharpened by the partial-warmup setup. Top-k OPD belongs in the toolkit as a stability or efficiency lever once the student and teacher already share enough local support. It is not a cold-start method, and a brief full-vocabulary phase does not bridge the gap on its own. The widest tested truncation, $k=4$, is the least destructive of the three, but it still falls well short of the full-vocabulary signal.

## Scope

What this run establishes is narrow: in the cold-start toy setting, 100 full-vocabulary steps is not enough warmup to stabilize student-top-k truncation. It does not locate where enough would be. The natural follow-up is to sweep `warmup_steps` more thoroughly and look for the threshold, measured either on top-1 agreement or on teacher mass over the student's selected support, above which top-k becomes stable. That sweep is the next run in the appendix.
