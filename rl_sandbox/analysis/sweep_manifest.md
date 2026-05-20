# Evidence sweep manifest

The exact commands that produced the tables in [`results_matrix.md`](results_matrix.md). Run from the repository root. The trainer picks CUDA automatically when PyTorch can see a GPU.

Sweeps are intentionally small: three seeds, `batch_size=96`, short toy-task horizons. Treat them as regime checks rather than benchmark claims.

Regenerate figures after the sweeps complete:

```bash
python rl_sandbox/analysis/plot_evidence.py
```

## Token-reversal influence baselines

```bash
python -m rl_sandbox.train --task token_reversal --method DG --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/dg_token.csv
python -m rl_sandbox.train --task token_reversal --method GRPO --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/grpo_token.csv
python -m rl_sandbox.train --task token_reversal --method TPO --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/tpo_token.csv
```

## Reward-noise robustness

All commands use `--reward_noise 0.2 --reward_noise_mode false_positive_rare_token`.

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

The 1500-step runs are the evidence runs. Short 300-step probes were under-budget for exact-match `chain_reversal`.

```bash
python -m rl_sandbox.train --task chain_reversal --method CE --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_ce_1500.csv
python -m rl_sandbox.train --task chain_reversal --method SelfDistillDG --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_selfdistilldg_1500.csv
python -m rl_sandbox.train --task chain_reversal --method SCOPELite --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3 --output results/chain_scopelite_1500.csv
```

## Freshness-aware replay

Capacity 5 at delay 4 is fixed-age stale replay. Capacity 32 is the stale-buffer stress test.

```bash
python -m rl_sandbox.train --task token_reversal --method DG --batch_size 96 --delay 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_dg_delay4.csv
python -m rl_sandbox.train --task token_reversal --method ReplayDG --batch_size 96 --delay 4 --replay_capacity 5 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_replaydg_cap5_delay4.csv
python -m rl_sandbox.train --task token_reversal --method FreshDG --batch_size 96 --delay 4 --replay_capacity 5 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_freshdg_cap5_delay4.csv
python -m rl_sandbox.train --task token_reversal --method ReplayDG --batch_size 96 --delay 4 --replay_capacity 32 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_replaydg_delay4.csv
python -m rl_sandbox.train --task token_reversal --method FreshDG --batch_size 96 --delay 4 --replay_capacity 32 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_freshdg_delay4.csv
python -m rl_sandbox.train --task token_reversal --method FreshDG --batch_size 96 --delay 4 --replay_capacity 32 --replay_age_decay 0.5 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/replay_freshdg_decay05_delay4.csv
```

## Masked-reversal partial credit

```bash
python -m rl_sandbox.train --task masked_reversal --method CE --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_ce.csv
python -m rl_sandbox.train --task masked_reversal --method DG --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_dg.csv
python -m rl_sandbox.train --task masked_reversal --method DGToken --batch_size 96 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_dgtoken.csv
python -m rl_sandbox.train --task masked_reversal --method TEMPO --batch_size 96 --group_size 8 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_tempo.csv
python -m rl_sandbox.train --task masked_reversal --method TPOToken --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_tpotoken.csv
python -m rl_sandbox.train --task masked_reversal --method GRPOToken --batch_size 96 --group_size 8 --inner_epochs 4 --num_steps 300 --eval_every 20 --num_seeds 3 --output results/masked_axis_grpotoken.csv
```

## Entropy-collapse diagnostics

```bash
python -m rl_sandbox.train --task token_reversal --method DG --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_dg.csv
python -m rl_sandbox.train --task token_reversal --method DGEntropyGuard --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_dgentropyguard.csv
python -m rl_sandbox.train --task token_reversal --method ASPO --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_aspo.csv
python -m rl_sandbox.train --task token_reversal --method R2VPO --batch_size 96 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_r2vpo.csv
python -m rl_sandbox.train --task token_reversal --method GRPO --batch_size 96 --group_size 8 --inner_epochs 4 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_grpo.csv
python -m rl_sandbox.train --task token_reversal --method TPO --batch_size 96 --group_size 8 --inner_epochs 4 --entropy_diagnostics true --num_steps 300 --eval_every 20 --num_seeds 3 --output results/entropy_tpo.csv
```
