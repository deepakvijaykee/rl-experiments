# Results Matrix

Compact GPU sweeps. Three seeds, `batch_size=96`, default token-reversal model unless noted. Treat these as regime checks. The absolute numbers move with seed and horizon; the ordering between methods and the shape of the failure modes do not. Each section below has the run command, the resulting numbers, and a short reading of what I take from them. The top-level README carries the synthesis; this document is the source data the synthesis draws from.

Reproduction commands are in [`sweep_manifest.md`](sweep_manifest.md). Figures (including those embedded in the top-level README) regenerate with `python rl_sandbox/analysis/plot_evidence.py`.

## Token reversal: clean influence baselines

![Clean token-reversal learning curves](figures/influence.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

| Method | Extra args | Final `test_error` |
| --- | --- | --- |
| `DG` | none | `0.3345 +/- 0.0043` |
| `GRPO` | `--group_size 8 --inner_epochs 4` | `0.3536 +/- 0.0106` |
| `TPO` | `--group_size 8 --inner_epochs 4` | `0.2399 +/- 0.0573` |

`TPO` is the strongest in this compact run, and the gap survives the seed variance. The entropy sweep below rules out the simplest alternative explanation (faster entropy collapse), which leaves the candidate-target construction as the working hypothesis for what is doing the work. The point I draw from the table on its own is more limited: if you already pay for grouped rollouts, the target-construction step is essentially free and on this task it carries the gain over `GRPO`-style use of the same rollouts.

## Reward-noise robustness

![False-positive reward-noise learning curves](figures/reward_noise.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --reward_noise 0.2 --reward_noise_mode false_positive_rare_token \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

| Method | Extra args | Final `test_error` | Final entropy |
| --- | --- | --- | --- |
| `DG` | none | `0.3778 +/- 0.0034` | `0.9599 +/- 0.0510` |
| `UncertaintyDG` | none | `0.4013 +/- 0.0286` | `1.0052 +/- 0.0643` |
| `FilteredDG` | default threshold `0.5` | `0.3778 +/- 0.0034` | `0.9599 +/- 0.0510` |
| `FilteredDG` | `--uncertainty_threshold 0.2` | `0.6536 +/- 0.1260` | `1.0322 +/- 0.0297` |
| `FilteredDG` | `--uncertainty_threshold 0.3` | `0.3810 +/- 0.0129` | `0.9883 +/- 0.0647` |
| `RewardVarianceDG` | none | `0.3921 +/- 0.0209` | `0.7879 +/- 0.0360` |
| `ASPO` | none | `0.3708 +/- 0.0095` | `0.1531 +/- 0.0589` |
| `R2VPO` | none | `0.3787 +/- 0.0042` | `0.1811 +/- 0.0428` |

The pattern across the table is that the methods which look most robust to noise also reach the lowest entropy. `ASPO` and `R2VPO` are the clearest cases, with small final-error gaps to `DG` but very low final entropy. The conservative gates (`UncertaintyDG`, `RewardVarianceDG`) keep entropy intact but pay in speed. `FilteredDG` is a cautionary tale about proxy granularity: its batch-level uncertainty signal in ungrouped runs collapses the method into all-or-nothing thresholding, which is why `--uncertainty_threshold 0.2` drops every batch. What I take from this table is that on this task, "noise robustness" is mostly tracking regularization strength. A robustness comparison that omits final entropy is reporting half the story, because the regularization that buys robustness also costs exploration.

## Reward-chain dense correction

![Reward-chain dense correction trajectories](figures/dense_correction.png)

```bash
python -m rl_sandbox.train --task chain_reversal --batch_size 96 \
  --eval_every 50 --num_seeds 3
```

At 300 steps, exact-match `test_error` is too strict to judge the task: even oracle `CE` finishes near `0.9867`. At 1500 steps the task is learnable.

| Method | Steps | Final `test_error` | First zero-error step |
| --- | ---: | --- | --- |
| `CE` | `1500` | `0.0000 +/- 0.0000` | `733.3333 +/- 57.7350` |
| `SelfDistillDG` | `1500` | `0.0000 +/- 0.0000` | `466.6667 +/- 125.8306` |
| `SCOPELite` | `1500` | `0.0000 +/- 0.0000` | `533.3333 +/- 104.0833` |

Both `SelfDistillDG` and `SCOPELite` reach zero exact-match error before `CE` does. The earlier weak `SCOPELite` reading I had at 300 steps was an under-budget artifact rather than a method failure; with 1500 steps the dense-correction path clearly works. What I would chase up at scale is the 200-300 step speedup over `CE` (`SelfDistillDG` first hits zero around step 467 versus `CE` at step 733). If that gap holds with a learned reviser instead of an oracle label, dense correction becomes a usable lever for sparse-reward chain tasks beyond the sandbox.

## Freshness-aware replay

![Replay freshness trajectories](figures/replay.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --delay 4 --num_steps 300 --eval_every 20 --num_seeds 3
```

| Method | Replay args | Final `test_error` | Mean replay age |
| --- | --- | --- | --- |
| `DG` | none | `0.5032 +/- 0.0080` | n/a |
| `ReplayDG` | `--replay_capacity 5` | `0.5032 +/- 0.0080` | `4.0000 +/- 0.0000` |
| `FreshDG` | `--replay_capacity 5` | `0.4902 +/- 0.0049` | `4.0000 +/- 0.0000` |
| `ReplayDG` | `--replay_capacity 32` | `0.9997 +/- 0.0006` | `18.0667 +/- 9.5237` |
| `FreshDG` | `--replay_capacity 32` | `0.5946 +/- 0.1391` | `16.4667 +/- 7.9418` |
| `FreshDG` | `--replay_capacity 32 --replay_age_decay 0.5` | `0.5048 +/- 0.0082` | `7.4444 +/- 5.5496` |

Two regimes, and they tell different stories. Capacity 5 at delay 4 is fixed-age stale replay: every sample is exactly 4 steps old, and `FreshDG` is only marginally more stable than delayed `DG` because the freshness weighting has nothing meaningful to weigh against. Capacity 32 is the interesting case, a stale-buffer stress test where the mean sample age is roughly 16-18 steps, well past the nominal delay. Without freshness decay the buffer collapses entropy, because the gradient averages over policies the current one no longer resembles. With strong decay (`--replay_age_decay 0.5`) the effective age halves and the method recovers. What I read from this table is that buffer capacity is a misleading control variable. The variable that determines whether replay helps is the effective sample age distribution, and any replay comparison that does not report it is reporting the wrong axis.

## Masked reversal: partial credit

![Masked-reversal scored and unscored trajectories](figures/partial_credit.png)

```bash
python -m rl_sandbox.train --task masked_reversal --batch_size 96 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

Grouped and token-candidate methods additionally use `--group_size 8`; token-candidate methods use `--inner_epochs 4`.

| Method | Extra args | Scored `test_error` | Unscored `test_error` |
| --- | --- | --- | --- |
| `CE` | none | `0.2738 +/- 0.0612` | `0.2793 +/- 0.0087` |
| `DG` | none | `0.3260 +/- 0.0132` | `0.4711 +/- 0.0149` |
| `DGToken` | none | `0.2851 +/- 0.0234` | `0.6286 +/- 0.0392` |
| `TEMPO` | `--group_size 8` | `0.4975 +/- 0.0098` | `0.5018 +/- 0.0049` |
| `TPOToken` | `--group_size 8 --inner_epochs 4` | `0.0000 +/- 0.0000` | `0.4910 +/- 0.0133` |
| `GRPOToken` | `--group_size 8 --inner_epochs 4` | `0.1229 +/- 0.2126` | `0.4525 +/- 0.0560` |

Reading the table along both columns is what reveals the trade-offs; reading only the scored column would misrepresent every method here. `TPOToken` drives the scored suffix to zero error and leaves unscored positions at chance, which is the clean partial-credit result. `DGToken` improves the scored suffix relative to sequence-level `DG` but actively damages the unscored positions, because its return-to-go credit concentrates on scored positions. `TEMPO` is out of regime: all three seeds stop early once grouped rollouts no longer carry mixed-reward batches. What I take from this table is that token-level credit is a sharper tool than DG-style sequence credit, not a more powerful one. It routes the existing reward signal more precisely; it cannot recover signal where the reward function does not place any. Comparing token-level methods on scored-suffix accuracy alone misses where the methodological disagreement lives.

## Entropy-collapse diagnostics

![Entropy and accuracy trajectories](figures/entropy.png)

```bash
python -m rl_sandbox.train --task token_reversal --batch_size 96 \
  --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3
```

Grouped methods additionally use `--group_size 8 --inner_epochs 4`.

| Method | Extra args | Final `test_error` | Final entropy | First entropy `<= 0.1` |
| --- | --- | --- | --- | --- |
| `DG` | none | `0.3345 +/- 0.0043` | `0.4695 +/- 0.0243` | n/a |
| `DGEntropyGuard` | none | `0.3358 +/- 0.0064` | `0.4847 +/- 0.0219` | n/a |
| `ASPO` | none | `0.3742 +/- 0.0040` | `0.0174 +/- 0.0020` | `113.3333 +/- 11.5470` |
| `R2VPO` | none | `0.3754 +/- 0.0023` | `0.0244 +/- 0.0087` | `106.6667 +/- 11.5470` |
| `GRPO` | `--group_size 8 --inner_epochs 4` | `0.3536 +/- 0.0106` | `0.0488 +/- 0.0305` | `53.3333 +/- 11.5470` |
| `TPO` | `--group_size 8 --inner_epochs 4` | `0.2399 +/- 0.0573` | `0.4057 +/- 0.0824` | n/a |

| Method | Whole-run `batch_delta_entropy` | Early `batch_delta_entropy` | Whole-run positive-advantage entropy drop |
| --- | --- | --- | --- |
| `DG` | `-0.0020 +/- 0.0069` | `-0.0041 +/- 0.0096` | `0.0034 +/- 0.0075` |
| `DGEntropyGuard` | `-0.0020 +/- 0.0080` | `-0.0037 +/- 0.0107` | `0.0027 +/- 0.0069` |
| `ASPO` | `-0.0029 +/- 0.0092` | `-0.0073 +/- 0.0120` | `0.1067 +/- 0.1761` |
| `R2VPO` | `-0.0029 +/- 0.0104` | `-0.0067 +/- 0.0141` | `0.0983 +/- 0.1682` |
| `GRPO` | `-0.0070 +/- 0.0234` | `-0.0090 +/- 0.0263` | `0.0593 +/- 0.0957` |
| `TPO` | `0.0026 +/- 0.0208` | `-0.0011 +/- 0.0215` | `0.0049 +/- 0.0187` |

The first-entropy-below-0.1 column is the most direct measurement. `GRPO` crosses that threshold around step 53, `R2VPO` and `ASPO` near step 100, and `TPO` and the DG family never cross it inside the 300-step horizon. The second table breaks down where the entropy goes. `GRPO`'s whole-run delta is the most negative, and its positive-advantage entropy drop is large despite a modest average advantage. That is consistent with within-group standardization amplifying small per-rollout differences into large effective gradients. `ASPO` and `R2VPO` show large positive-advantage entropy drops without a correspondingly large whole-run delta, meaning their entropy loss concentrates on a subset of steps and is policy-collapse-shaped rather than gradual decay. What I take from these two tables together is that `GRPO`'s entropy collapse is driven by the within-group normalization, and the PPO clip sits downstream of the amplification, so it does not constrain the thing causing the collapse. That is also why `DGEntropyGuard` helps only marginally on this task: it operates at the sampled-action probability level, again downstream of the same amplification. If I were going to design an entropy fix for `GRPO`-family methods, the normalization is where I would start. The clip is not the constraint that is failing.
