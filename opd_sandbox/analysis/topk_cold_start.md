# Top-K Cold Start

This appendix run tests the natural follow-up to the top-k stability result:
if top-k reverse KL fails from a random cold start because support overlap is
too weak, does a short full-vocabulary OPD warmup make top-k stable?

The experiment compares:

- `full_vocab_rkl`: full-vocabulary reverse KL for all 300 steps.
- `cold_topk_k{1,2,4}`: student-top-k truncated reverse KL from step 0.
- `warm_topk_k{1,2,4}`: full-vocabulary reverse KL for 100 steps, then the
  matching top-k truncated objective for the remaining 200 steps.

The 100-step warmup is a probe, not a recommended schedule. It is long enough
to create visible alignment and short enough that the switch to top-k can still
stress the support-overlap assumption.

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

Outputs:

- `opd_sandbox/analysis/results/topk_cold_start.csv`
- `opd_sandbox/analysis/results/topk_cold_start.png`
- per-variant CSVs in the same directory

The evidence run completed in about 106 seconds on the local machine.

## Result

Final greedy evaluation at step 300:

| Variant | Test error | Entropy |
| --- | ---: | ---: |
| `full_vocab_rkl` | 0.0010 +/- 0.0012 | 0.0608 +/- 0.0227 |
| `cold_topk_k1` | 0.8670 +/- 0.0136 | 2.1971 +/- 0.0001 |
| `cold_topk_k2` | 0.8666 +/- 0.0190 | 2.1972 +/- 0.0001 |
| `cold_topk_k4` | 0.8448 +/- 0.0230 | 2.1972 +/- 0.0000 |
| `warm_topk_k1` | 0.7981 +/- 0.0166 | 2.1969 +/- 0.0001 |
| `warm_topk_k2` | 0.7749 +/- 0.0440 | 2.1971 +/- 0.0001 |
| `warm_topk_k4` | 0.6963 +/- 0.0166 | 1.9873 +/- 0.0185 |

Last logged OPD diagnostics at step 280:

| Variant | Reverse KL | Top-1 agreement | Overlap@4 | Reward |
| --- | ---: | ---: | ---: | ---: |
| `full_vocab_rkl` | 0.6645 +/- 0.5742 | 0.9355 +/- 0.0609 | 0.5415 +/- 0.0299 | 0.8971 +/- 0.0827 |
| `cold_topk_k1` | 5.7899 +/- 0.0014 | 0.1432 +/- 0.0158 | 0.4508 +/- 0.0456 | 0.1094 +/- 0.0220 |
| `cold_topk_k2` | 5.7906 +/- 0.0006 | 0.1491 +/- 0.0079 | 0.4316 +/- 0.0220 | 0.1107 +/- 0.0209 |
| `cold_topk_k4` | 5.7904 +/- 0.0002 | 0.1549 +/- 0.0041 | 0.4740 +/- 0.0421 | 0.1100 +/- 0.0214 |
| `warm_topk_k1` | 5.7837 +/- 0.0013 | 0.1940 +/- 0.0502 | 0.5099 +/- 0.0698 | 0.1107 +/- 0.0209 |
| `warm_topk_k2` | 5.7832 +/- 0.0025 | 0.2389 +/- 0.0127 | 0.4468 +/- 0.0630 | 0.1113 +/- 0.0220 |
| `warm_topk_k4` | 5.1169 +/- 0.0304 | 0.2747 +/- 0.0285 | 0.5535 +/- 0.0249 | 0.2083 +/- 0.0108 |

`Overlap@4` is tie-sensitive in this oracle task because all wrong teacher
tokens have equal probability. Treat it as a rough geometry check, not as the
main switch criterion. Top-1 agreement and reward carry the cleaner signal.

Trajectory means show the failure mode clearly:

| Variant | Step 100 test error | Step 120 test error | Step 280 test error |
| --- | ---: | ---: | ---: |
| `full_vocab_rkl` | 0.6243 | 0.5978 | 0.0574 |
| `warm_topk_k1` | 0.6243 | 0.7060 | 0.8079 |
| `warm_topk_k2` | 0.6243 | 0.6983 | 0.7575 |
| `warm_topk_k4` | 0.6243 | 0.6306 | 0.7118 |

All warm variants share the same full-vocabulary trajectory through step 100.
After the switch, `k=1` and `k=2` degrade almost immediately. `k=4` degrades
more slowly and retains some improvement, but it still moves far away from the
full-vocabulary baseline.

## Interpretation

The 100-step warmup creates partial overlap, but not enough overlap to make
student-top-k truncation stable. At the switch point, full-vocabulary OPD has
only reached about 0.37 top-1 agreement and 0.29 sampled reward. Once the
objective is restricted to the student's current top-k support, the missing
teacher-preferred tokens again stop receiving reliable gradient.

The important result is not merely "warmup helps a bit." It is sharper:

> A partial full-vocabulary warmup can be erased by switching too early to
> top-k truncation.

The switch itself is an intervention. If it removes omitted-token signal before
the behavior is anchored, the top-k phase can undo the asymmetry that the warmup
just created.

This matches the mechanism suggested by the previous support-overlap run. Top-k
OPD is not a cold-start method; it is a stability/efficiency method for a
regime where teacher and student already share enough local support. In this
toy setup, a 100-step warmup does not reach that regime. The widest tested
truncation, `k=4`, is less destructive than `k=1` or `k=2`, but still not a
faithful substitute for the full-vocabulary signal.

## Scope

This run does not prove that top-k OPD fails after any warmup. It only shows
that a short partial warmup is insufficient in this cold-start toy setting.
A natural next probe is to vary `warmup_steps` and ask whether there is a
threshold of top-1 agreement or teacher mass-on-student-support above which
top-k becomes stable.
