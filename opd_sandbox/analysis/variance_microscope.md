# Variance Microscope

This is the first OPD appendix result. It is a mechanism check for estimator
variance, not a benchmark and not a faithful MiniLLM reproduction.

The fixed random transformer is deliberate. It removes convergence quality,
teacher quality, and optimizer schedule from the question, leaving only the
pre-optimization question: how much horizon-dependent noise does each
reverse-KL estimator inject before learning has a chance to help?

The experiment fixes a tiny random transformer and repeatedly samples
student rollouts from `ReversalTask`. On each batch it computes three
OPD-style gradients on the same visited prefixes:

| Estimator | What it measures |
| --- | --- |
| `sequence_pg` | Sampled cumulative-return score estimator, using `R_t = sum_{u >= t} r_u`. |
| `token_pg` | Sampled one-step score estimator with gamma=0 credit. |
| `full_vocab_rkl` | Exact per-token `KL(pi_student || pi_teacher)` over the full vocabulary. |

For each estimator, the script records the variance of a fixed random
projection of the gradient across rollout batches. This avoids comparing only
gradient norms, which mix estimator scale and estimator noise.

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

The sequence-level estimator is already about 398x noisier than exact
full-vocabulary reverse KL at horizon 4. By horizon 64, that ratio is about
1.28e7x. The one-step sampled estimator stays within roughly 4-5x of exact
reverse KL across the same sweep.

## Interpretation

The result shows the optimization pathology the OPD notes are trying to make
concrete: when each token receives a cumulative sampled return, the gradient
variance grows rapidly with horizon. In this setup the growth is steeper than
quadratic, which is expected because the estimator sums score terms whose
returns also grow with horizon.

The per-token estimators are deliberately averaged per token, matching the
appendix sandbox's training convention. Under that normalization, `token_pg`
and `full_vocab_rkl` become stable with longer horizons because each batch
contains more token-level averaging. If those objectives were rescaled back to
a raw sequence-sum objective, their variance curves would shift upward by the
corresponding length factor. The qualitative comparison remains the point:
cumulative sampled returns are the unstable part.

I would not read the fitted slopes as universal scaling exponents. They depend
on the toy teacher, model initialization, batch construction, and exact
normalization. The robust result is the ordering: cumulative-return sampled
gradients accumulate horizon-dependent noise much faster than local per-token
estimators.

`full_vocab_rkl` is the cleanest estimator because it does not sample the token
inside the KL term. `token_pg` keeps the on-policy sampled-token form, so it is
noisier, but it avoids the cumulative-return amplification. This is the toy
version of why practical OPD implementations prefer per-token KL or aggressive
variance reduction over naive sequence-level REINFORCE.

## Scope

This result supports a narrow claim:

> In a controlled OPD toy setting, cumulative sampled-return gradients have
> much worse horizon scaling than per-token OPD gradients.

It does not claim a reproduction of MiniLLM, GKD, Tinker, or any frontier-scale
OPD result. There is no learned teacher, no mixed teacher/student sampling, no
top-k truncation, no long-CoT task, and no optimization-to-convergence claim.
The fixed random model is intentional: the experiment isolates estimator
variance before training dynamics add confounds.
