# Evidence sweep manifest

The exact commands that produced the tables in [`results_matrix.md`](results_matrix.md). Run from the repository root. The trainer picks CUDA automatically when PyTorch can see a GPU.

Each sweep is small on purpose: three seeds, `batch_size=96`, and short toy-task horizons. They are regime checks and not benchmark claims, so what each table is good for is the ordering between methods and the shape of each failure mode. The absolute numbers move with seed and horizon and carry much less.

Every section below states why its configuration is what it is. The commands are only half of a reproduction, since a sweep run at the wrong horizon or the wrong buffer capacity reproduces the syntax and not the experiment, and two of the sweeps here were originally read wrong for exactly that reason.

Regenerate the figures after the sweeps complete:

```bash
python rl_sandbox/analysis/plot_evidence.py
```

## Token-reversal influence baselines

The clean-reward reference point that every other sweep is read against. Grouped methods get `--group_size 8 --inner_epochs 4` so that GRPO and TPO see identical rollout budgets, which is the condition that makes the information-per-group comparison meaningful. Changing either one turns it into a compute comparison instead.

```bash
python -m rl_sandbox.train --task token_reversal --method DG --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/dg_token.csv
python -m rl_sandbox.train --task token_reversal --method GRPO --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/grpo_token.csv
python -m rl_sandbox.train --task token_reversal --method TPO --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/tpo_token.csv
```

## Reward-noise robustness

All commands use `--reward_noise 0.2 --reward_noise_mode false_positive_rare_token`. The noise mode matters as much as the rate: false positives on rare tokens are the failure a verifier or judge actually produces, and a symmetric noise model would let a method look robust by simply averaging the noise away. The three `FilteredDG` thresholds are in the manifest because the method's behavior is threshold-dominated, and reporting only the default would hide that the proxy has degenerated into a batch-level switch.

```bash
python -m rl_sandbox.train --task token_reversal --method DG --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_dg_false_positive_rare.csv
python -m rl_sandbox.train --task token_reversal --method UncertaintyDG --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_uncertaintydg_false_positive_rare.csv
python -m rl_sandbox.train --task token_reversal --method FilteredDG --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_filtereddg_false_positive_rare.csv
python -m rl_sandbox.train --task token_reversal --method FilteredDG --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --uncertainty_threshold 0.2 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_filtereddg_thr02_false_positive_rare.csv
python -m rl_sandbox.train --task token_reversal --method FilteredDG --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --uncertainty_threshold 0.3 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_filtereddg_thr03_false_positive_rare.csv
python -m rl_sandbox.train --task token_reversal --method RewardVarianceDG --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_rewardvariancedg_false_positive_rare.csv
python -m rl_sandbox.train --task token_reversal --method ASPO --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_aspo_false_positive_rare.csv
python -m rl_sandbox.train --task token_reversal --method R2VPO --batch_size 96 --reward_noise 0.2 --reward_noise_mode false_positive_rare_token --num_steps 300 --eval_every 20 --num_seeds 3 --output results/noise_r2vpo_false_positive_rare.csv
```

## Reward-chain dense correction

1500 steps is the evidence horizon. The shorter 300-step probes are under-budget for exact-match `chain_reversal`: even the supervised baseline does not finish in time, so no method has room to demonstrate dense correction at the shorter horizon.

```bash
python -m rl_sandbox.train --task chain_reversal --method CE --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_ce_1500.csv
python -m rl_sandbox.train --task chain_reversal --method SelfDistillDG --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_selfdistilldg_1500.csv
python -m rl_sandbox.train --task chain_reversal --method SCOPELite --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_scopelite_1500.csv
```

## Freshness-aware replay

Capacity 5 at delay 4 is the fixed-age regime, where every sample in the buffer is exactly 4 steps old. Capacity 32 is the stress test that decouples capacity from sample age, letting the age distribution spread.

```bash
python -m rl_sandbox.train --task token_reversal --method DG --batch_size 96 --delay 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_dg_delay4.csv
python -m rl_sandbox.train --task token_reversal --method ReplayDG --batch_size 96 --delay 4 --replay_capacity 5 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_replaydg_cap5_delay4.csv
python -m rl_sandbox.train --task token_reversal --method FreshDG --batch_size 96 --delay 4 --replay_capacity 5 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_freshdg_cap5_delay4.csv
python -m rl_sandbox.train --task token_reversal --method ReplayDG --batch_size 96 --delay 4 --replay_capacity 32 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_replaydg_delay4.csv
python -m rl_sandbox.train --task token_reversal --method FreshDG --batch_size 96 --delay 4 --replay_capacity 32 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_freshdg_delay4.csv
python -m rl_sandbox.train --task token_reversal --method FreshDG --batch_size 96 --delay 4 --replay_capacity 32 --replay_age_decay 0.5 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_freshdg_decay05_delay4.csv
```

## Masked-reversal partial credit

Both the scored and unscored error columns have to be logged for this sweep to mean anything, since the scored column alone makes every method look successful. Grouped methods take `--group_size 8` and the token-candidate methods additionally take `--inner_epochs 4`, matching the influence sweep so the two are readable side by side.

```bash
python -m rl_sandbox.train --task masked_reversal --method CE --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_ce.csv
python -m rl_sandbox.train --task masked_reversal --method DG --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_dg.csv
python -m rl_sandbox.train --task masked_reversal --method DGToken --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_dgtoken.csv
python -m rl_sandbox.train --task masked_reversal --method TEMPO --batch_size 96 --group_size 8 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_tempo.csv
python -m rl_sandbox.train --task masked_reversal --method TPOToken --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_tpotoken.csv
python -m rl_sandbox.train --task masked_reversal --method GRPOToken --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_grpotoken.csv
```

## Entropy-collapse diagnostics

`--entropy_diagnostics true` is what turns on the per-step entropy decomposition, without which the collapse is only visible after the fact in the final entropy value. The horizon matters here more than elsewhere: GRPO crosses the 0.1 entropy threshold around step 53, so a sweep shorter than roughly 100 steps would record the collapse without ever showing the flat test-error curve that follows it.

```bash
python -m rl_sandbox.train --task token_reversal --method DG --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_dg.csv
python -m rl_sandbox.train --task token_reversal --method DGEntropyGuard --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_dgentropyguard.csv
python -m rl_sandbox.train --task token_reversal --method ASPO --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_aspo.csv
python -m rl_sandbox.train --task token_reversal --method R2VPO --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_r2vpo.csv
python -m rl_sandbox.train --task token_reversal --method GRPO --batch_size 96 --group_size 8 --inner_epochs 4 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_grpo.csv
python -m rl_sandbox.train --task token_reversal --method TPO --batch_size 96 --group_size 8 --inner_epochs 4 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_tpo.csv
```
