# Results matrix

The runs collected here are compact GPU sweeps: three seeds, `batch_size=96`, the default token-reversal model unless noted. Absolute numbers move with seed and horizon, but the ordering between methods and the qualitative shape of each failure mode have been stable across the variations I checked. The top-level README pulls these six tables into a single narrative. This document is the source data behind that narrative. Each section gives the run command, the resulting table, and an immediate reading of what the table says.

Reproduction commands live in [`sweep_manifest.md`](sweep_manifest.md). The figures the top-level README embeds regenerate with `python rl_sandbox/analysis/plot_evidence.py`.

## Token reversal: clean influence baselines

![Clean token-reversal learning curves](figures/influence.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

| Method | Extra args | Final `test_error` |
| --- | --- | --- |
| `DG` | none | `0.3345 ± 0.0043` |
| `GRPO` | `--group_size 8 --inner_epochs 4` | `0.3536 ± 0.0106` |
| `TPO` | `--group_size 8 --inner_epochs 4` | `0.2399 ± 0.0573` |

TPO finishes ahead of GRPO on this run by more than the seed-to-seed variance, and the entropy diagnostics in the table at the bottom of this document rule out the most natural counter-explanation: TPO keeps final entropy near DG's while GRPO crosses the 0.1 threshold around step 53. The gap is not faster collapse. It is more information used per gradient step. With eight rollouts per group, GRPO reads the group as a noise-reduction device for the advantage and applies that advantage to the one action that was actually sampled, producing a single weighted gradient direction per group. TPO reads the same eight rollouts as candidates for the update target, builds a soft target distribution over them weighted by relative rewards, and pulls the policy toward that target. The construction adds K weighted gradient directions per group rather than one, at negligible additional compute on top of rollouts that have already been paid for. On this task the information-per-step difference shows up as the test-error gap.

## Reward-noise robustness

![False-positive reward-noise learning curves](figures/reward_noise.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --reward_noise 0.2 --reward_noise_mode false_positive_rare_token \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

| Method | Extra args | Final `test_error` | Final entropy |
| --- | --- | --- | --- |
| `DG` | none | `0.3778 ± 0.0034` | `0.9599 ± 0.0510` |
| `UncertaintyDG` | none | `0.4013 ± 0.0286` | `1.0052 ± 0.0643` |
| `FilteredDG` | default threshold `0.5` | `0.3778 ± 0.0034` | `0.9599 ± 0.0510` |
| `FilteredDG` | `--uncertainty_threshold 0.2` | `0.6536 ± 0.1260` | `1.0322 ± 0.0297` |
| `FilteredDG` | `--uncertainty_threshold 0.3` | `0.3810 ± 0.0129` | `0.9883 ± 0.0647` |
| `RewardVarianceDG` | none | `0.3921 ± 0.0209` | `0.7879 ± 0.0360` |
| `ASPO` | none | `0.3708 ± 0.0095` | `0.1531 ± 0.0589` |
| `R2VPO` | none | `0.3787 ± 0.0042` | `0.1811 ± 0.0428` |

Either column on its own tells a misleading story. The methods that reach the lowest final error in this sweep, ASPO and R2VPO, are also the methods that reach the lowest final entropy by a wide margin (0.15 and 0.18 against DG's 0.96), which means they paid for the error advantage in burned exploration. The conservative gates UncertaintyDG and RewardVarianceDG take the opposite trade. They preserve entropy near DG's level and accept slower learning instead.

FilteredDG is the case where the proxy itself becomes the confound. In the ungrouped runs used here, its uncertainty signal is computed at the batch level, so the threshold sees a single value across the whole batch and ends up acting as all-or-nothing thresholding rather than per-sample filtering. At threshold `0.5` essentially every batch passes and the method reproduces the no-filter baseline. At threshold `0.2` essentially every batch is dropped and the optimizer has almost no gradient signal to learn from. Test error climbs from 0.38 to 0.65 with much wider variance, which is what one would expect from an optimizer trying to learn from a near-empty gradient. The failure is the proxy granularity, not the threshold tuning.

What the full table says, read both columns together, is that "robustness" on this task is tracking regularization strength. Any comparison that reports a robustness number without the corresponding final entropy is reporting half of the operating point each method is choosing, and the half it is leaving out is the cost side of the trade.

## Reward-chain dense correction

![Reward-chain dense correction trajectories](figures/dense_correction.png)

```bash
python -m rl_sandbox.train --task chain_reversal --batch_size 96 \
  --eval_every 50 --num_seeds 3
```

At 300 steps, exact-match `test_error` is too strict to discriminate between methods on this task. Even supervised CE finishes near 0.99 error, so no reviser-based method has the budget to demonstrate dense correction. 1500 steps is the horizon at which the question becomes answerable.

| Method | Steps | Final `test_error` | First zero-error step |
| --- | ---: | --- | --- |
| `CE` | `1500` | `0.0000 ± 0.0000` | `733.3333 ± 57.7350` |
| `SelfDistillDG` | `1500` | `0.0000 ± 0.0000` | `466.6667 ± 125.8306` |
| `SCOPELite` | `1500` | `0.0000 ± 0.0000` | `533.3333 ± 104.0833` |

Given enough steps, both reviser-based methods reach zero exact-match error before supervised CE does. SelfDistillDG first hits zero around step 467, SCOPELite around step 533, and CE not until step 733. The weak SCOPELite reading I had reported at 300 steps was an under-budget artifact rather than a method failure: at that horizon the oracle baseline had not finished yet, so there was no room for any method to show its speedup. The result worth chasing at scale is the 200- to 300-step speedup of SelfDistillDG over CE. The oracle reviser is the load-bearing element of the toy version, and the natural next question is whether the speedup survives substitution by a learned, noisier reviser. That is what would turn dense correction into a usable lever for sparse-reward chain tasks rather than a property of having an oracle in the loop.

## Freshness-aware replay

![Replay freshness trajectories](figures/replay.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --delay 4 --num_steps 300 --eval_every 20 --num_seeds 3
```

| Method | Replay args | Final `test_error` | Mean replay age |
| --- | --- | --- | --- |
| `DG` | none | `0.5032 ± 0.0080` | n/a |
| `ReplayDG` | `--replay_capacity 5` | `0.5032 ± 0.0080` | `4.0000 ± 0.0000` |
| `FreshDG` | `--replay_capacity 5` | `0.4902 ± 0.0049` | `4.0000 ± 0.0000` |
| `ReplayDG` | `--replay_capacity 32` | `0.9997 ± 0.0006` | `18.0667 ± 9.5237` |
| `FreshDG` | `--replay_capacity 32` | `0.5946 ± 0.1391` | `16.4667 ± 7.9418` |
| `FreshDG` | `--replay_capacity 32 --replay_age_decay 0.5` | `0.5048 ± 0.0082` | `7.4444 ± 5.5496` |

The table contains two regimes, and they put different questions to the methods. The capacity-5-at-delay-4 rows are a fixed-age regime by construction: every sample in the buffer is exactly 4 steps old, with no spread for any freshness-aware method to exploit. FreshDG is only marginally more stable than delayed DG in those rows because its age weights have nothing meaningful to compare across when every sample shares the same age.

The capacity-32 rows are the stress test that decouples capacity from age and lets the age distribution spread. Mean sample age climbs to roughly 16 to 18 steps, well past the nominal delay, and the averaged gradient ends up pointing at policies the current one no longer resembles. Without any freshness decay, that averaged direction is wrong often enough to collapse entropy and drive test error to chance. With `--replay_age_decay 0.5`, the effective sample age halves to around 7.4, the staleness bias drops back into the range the importance ratios can absorb, and the method recovers final error close to delayed-DG.

The reading I take is that capacity is the variance knob in replay's variance-vs-bias trade and age is the bias knob, and a comparison that reports buffer capacity alone is tuning one knob while leaving the other one implicit. Conclusions formed that way transfer poorly to setups where the implicit knob takes a different value, which is the warning the table is meant to deliver.

## Masked reversal: partial credit

![Masked-reversal scored and unscored trajectories](figures/partial_credit.png)

```bash
python -m rl_sandbox.train --task masked_reversal --batch_size 96 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

Grouped and token-candidate methods additionally use `--group_size 8`, and token-candidate methods use `--inner_epochs 4`.

| Method | Extra args | Scored `test_error` | Unscored `test_error` |
| --- | --- | --- | --- |
| `CE` | none | `0.2738 ± 0.0612` | `0.2793 ± 0.0087` |
| `DG` | none | `0.3260 ± 0.0132` | `0.4711 ± 0.0149` |
| `DGToken` | none | `0.2851 ± 0.0234` | `0.6286 ± 0.0392` |
| `TEMPO` | `--group_size 8` | `0.4975 ± 0.0098` | `0.5018 ± 0.0049` |
| `TPOToken` | `--group_size 8 --inner_epochs 4` | `0.0000 ± 0.0000` | `0.4910 ± 0.0133` |
| `GRPOToken` | `--group_size 8 --inner_epochs 4` | `0.1229 ± 0.2126` | `0.4525 ± 0.0560` |

Both columns carry signal, and reading only the scored column would misrepresent every method in the table. TPOToken drives the scored suffix to zero error while leaving unscored positions near chance, which is the cleanest possible per-position routing result: the credit is allocated where the reward function placed signal, and the method does not pretend to learn at positions the reward function declined to score. DGToken is the cautionary case. It improves the scored suffix relative to sequence-level DG, but it actively damages the unscored positions, because its return-to-go credit concentrates on positions that received reward and applies a smaller-than-baseline gradient elsewhere. TEMPO is out of regime entirely on this task. All three seeds stop training early once the grouped rollouts stop producing mixed-reward batches, which is the condition the prefix-tree credit implicitly assumes.

The mechanism behind these readings falls out of the token-level policy gradient itself. Each per-token update is proportional to `∇_θ log π(a_t | s_t) · A_t`, so at any position where the reward function places no signal the advantage `A_t` is zero, and no upstream sharpening of credit assignment can produce gradient where the multiplicative factor is zero. Token-level methods redistribute existing reward signal across positions. They cannot manufacture signal where the reward function placed none. Reading the scored column in isolation hides exactly that constraint, which is the part of the result an outside reader is most likely to misinterpret.

## Entropy-collapse diagnostics

![Entropy and accuracy trajectories](figures/entropy.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3
```

Grouped methods additionally use `--group_size 8 --inner_epochs 4`.

| Method | Extra args | Final `test_error` | Final entropy | First entropy `<= 0.1` |
| --- | --- | --- | --- | --- |
| `DG` | none | `0.3345 ± 0.0043` | `0.4695 ± 0.0243` | n/a |
| `DGEntropyGuard` | none | `0.3358 ± 0.0064` | `0.4847 ± 0.0219` | n/a |
| `ASPO` | none | `0.3742 ± 0.0040` | `0.0174 ± 0.0020` | `113.3333 ± 11.5470` |
| `R2VPO` | none | `0.3754 ± 0.0023` | `0.0244 ± 0.0087` | `106.6667 ± 11.5470` |
| `GRPO` | `--group_size 8 --inner_epochs 4` | `0.3536 ± 0.0106` | `0.0488 ± 0.0305` | `53.3333 ± 11.5470` |
| `TPO` | `--group_size 8 --inner_epochs 4` | `0.2399 ± 0.0573` | `0.4057 ± 0.0824` | n/a |

| Method | Whole-run `batch_delta_entropy` | Early `batch_delta_entropy` | Whole-run positive-advantage entropy drop |
| --- | --- | --- | --- |
| `DG` | `-0.0020 ± 0.0069` | `-0.0041 ± 0.0096` | `0.0034 ± 0.0075` |
| `DGEntropyGuard` | `-0.0020 ± 0.0080` | `-0.0037 ± 0.0107` | `0.0027 ± 0.0069` |
| `ASPO` | `-0.0029 ± 0.0092` | `-0.0073 ± 0.0120` | `0.1067 ± 0.1761` |
| `R2VPO` | `-0.0029 ± 0.0104` | `-0.0067 ± 0.0141` | `0.0983 ± 0.1682` |
| `GRPO` | `-0.0070 ± 0.0234` | `-0.0090 ± 0.0263` | `0.0593 ± 0.0957` |
| `TPO` | `0.0026 ± 0.0208` | `-0.0011 ± 0.0215` | `0.0049 ± 0.0187` |

The most direct measurement is the first table's last column: the step at which entropy first crosses 0.1. GRPO crosses it around step 53. ASPO and R2VPO near step 100. TPO and the DG family never cross it inside 300 steps at all. The second table decomposes where each method's entropy actually goes. GRPO's whole-run delta is the most negative, and its positive-advantage entropy drop is large despite the average advantage being modest, which is the signature the standardization mechanism predicts. The advantage formula `A_i = (R_i − μ) / σ` scales as `1 / σ` when the rollouts in a group mostly agree on the right answer, so even small per-rollout reward gaps produce large effective advantages, and every gradient step ends up saturating the trust region in the direction of whichever rollout happened to win the group.

ASPO and R2VPO show large positive-advantage entropy drops without a correspondingly large whole-run delta. Their entropy loss concentrates on a small fraction of steps, which is collapse-shaped rather than gradual decay. DGEntropyGuard barely moves the entropy needle in either table, and the reason is structural. It acts at the sampled-action probability level, downstream of the same standardization-driven amplification, and downstream is the wrong place to attack a problem whose source is upstream. The PPO clip sits in the same position relative to the standardization. The clip bounds how far one update can move the policy, but it does not bound the advantage that drives the update, and the advantage is the quantity the standardization has just amplified. Any practical fix for the entropy-collapse pathology has to act upstream of both the clip and the sampled-action guard. The variants DrGRPO (drop the σ in the advantage normalization) and DAPO (decouple the clipping range so positive and negative sides differ) are testing the prediction in two different ways.
