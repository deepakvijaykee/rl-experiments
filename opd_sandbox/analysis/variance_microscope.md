# Variance microscope

The first appendix experiment isolates a question that has to be settled before any of the later top-k comparisons make sense. Before learning is allowed to help, how much horizon-dependent noise does each reverse-KL estimator inject into the gradient? The rest of the OPD design space, including top-k truncation, warmup schedules, and teacher entropy, only starts to matter once the estimator itself produces a usable gradient signal. If the estimator is already drowning in horizon noise at short horizons, no downstream choice can recover from it.

To keep the answer clean, the experiment freezes the model. With a fixed random transformer in place of a training one, the variance number reflects the estimator alone, free of the optimization quality, teacher quality, and learning-schedule confounds that would otherwise creep in. The script samples student rollouts from `ReversalTask` against that fixed model and computes three OPD-style gradient estimators on the same visited prefixes.

| Estimator | What it measures |
| --- | --- |
| `sequence_pg` | Sampled cumulative-return score estimator, with `R_t = sum_{u >= t} r_u`. |
| `token_pg` | Sampled one-step score estimator with gamma=0 credit. |
| `full_vocab_rkl` | Exact per-token `KL(pi_student || pi_teacher)` over the full vocabulary. |

For each estimator, the script records the variance of a fixed random projection of the gradient across rollout batches. Projection variance is the right quantity to compare on; raw gradient norm would mix estimator scale with estimator noise, and the projection separates them.

## Command

Run from the repository root:

```bash
python -m opd_sandbox.experiments.variance_microscope \
  --horizons 4,8,16,32,64 \
  --num_batches 60 \
  --batch_size 16 \
  --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

Outputs:

- `opd_sandbox/analysis/results/variance_microscope.csv`
- `opd_sandbox/analysis/results/variance_microscope.png`

The evidence run completed in about 22 seconds on the local machine.

## Result

Mean gradient-projection variance across three seeds:

| Horizon | `full_vocab_rkl` | `token_pg` | `sequence_pg` |
| ---: | ---: | ---: | ---: |
| 4 | 1.19e-4 | 5.44e-4 | 4.73e-2 |
| 8 | 5.36e-5 | 2.32e-4 | 2.24e-1 |
| 16 | 2.91e-5 | 1.11e-4 | 1.75e+0 |
| 32 | 1.27e-5 | 5.64e-5 | 1.42e+1 |
| 64 | 6.93e-6 | 2.50e-5 | 8.89e+1 |

Log-log slope of projection variance versus horizon:

| Estimator | Slope | Growth from 4 to 64 |
| --- | ---: | ---: |
| `full_vocab_rkl` | -1.03 | 0.06x |
| `token_pg` | -1.09 | 0.05x |
| `sequence_pg` | 2.77 | 1882x |

The sequence-level estimator is already about 398x noisier than the exact full-vocabulary objective at horizon 4. By horizon 64 the ratio is roughly 1.28e7x. The one-step sampled estimator stays within 4 to 5x of the exact estimator across the same sweep.

## Interpretation

The cumulative-return pathology shows up in the slope, and the slope tells most of the story. When each token's score function carries a cumulative sampled return, both the magnitude of each summand and the number of summed terms grow with horizon. Gradient variance grows faster than the count of summands alone would predict, because each term in the sum is heavier-tailed at longer horizons. The score-function term at position `t` integrates more noisy samples through its return, so its variance contribution rises with horizon as well. The variance of the sum picks up an extra factor beyond the linear count of terms, which is why the fitted slope on `sequence_pg` is steeper than quadratic.

The per-token estimators in this sandbox average per token rather than summing across the sequence, matching the training convention used throughout the appendix. Under that normalization, `token_pg` and `full_vocab_rkl` actually become more stable with longer horizons, because each batch contains more per-token averaging. The ordering holds across normalization choices. If those estimators were rescaled back to a sum-over-tokens convention, their variance curves would shift upward by the corresponding length factor, but the ordering between the three estimators would not change. The cumulative sampled return is the unstable ingredient. The normalization only changes how the variance is distributed across the batch.

The fitted slopes are not universal scaling exponents. They depend on the toy teacher, the model initialization, the batch construction, and the exact normalization, and a different setup would produce different slopes. What survives those choices is the ordering: cumulative-return sampled gradients accumulate horizon-dependent noise much faster than local per-token estimators do.

`full_vocab_rkl` is the cleanest estimator of the three because it does not sample the token inside the KL term at all. The expectation is over the full vocabulary, so the variance contribution from sampling the action simply drops out. `token_pg` keeps the on-policy sampled-token form, which makes it noisier than the exact estimator, but it avoids the cumulative-return amplification. That mirrors why OPD pipelines reach for per-token KL or explicit variance reduction in preference to sequence-level REINFORCE.

## Scope

The robust thing to carry over from this toy is the ordering between estimators rather than the absolute slopes. The fixed random model is the experimental choice that keeps the ordering interpretable, because it isolates estimator variance from the optimization and teacher-quality confounds that would otherwise drift in. Mixed teacher-student sampling, top-k truncation, and long-CoT extensions each introduce another moving part, which is why this particular run holds all of them off.
