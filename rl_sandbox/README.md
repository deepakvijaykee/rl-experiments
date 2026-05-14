# rl_sandbox

The toy sandbox. Tasks small enough to inspect gradient flow directly, and a registry of update rules to compare on them.

Top-level context, install steps, headline findings, and figures live in the [repository README](../README.md). This page is the menu of methods and tasks, plus the experiment matrix I use to decide what to run.

## Quick runs

```bash
pip install -r requirements.txt

python -m rl_sandbox.train --method DG --delay 30
python -m rl_sandbox.train --task mnist --method TPOFullAction
python -m rl_sandbox.train --task token_reversal --method GRPO --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method TPO --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method TPOToken --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method DAPOLite --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method FreshDG --delay 4 --replay_capacity 5
python -m rl_sandbox.train --task chain_reversal --method SelfDistillDG
python -m rl_sandbox.train --task chain_arithmetic --method DG --seq_len 3 --vocab_size 5
python -m rl_sandbox.train --task format_answer --method DG --seq_len 3 --vocab_size 5
python -m rl_sandbox.train --task lm_bandit --method DG --model_name distilgpt2 --batch_size 16 --lr 5e-5
python -m rl_sandbox.train --sweep
python -m rl_sandbox.plot results.csv
```

Reproduction commands and result tables live in [`analysis/sweep_manifest.md`](analysis/sweep_manifest.md) and [`analysis/results_matrix.md`](analysis/results_matrix.md).

## Methods

Grouped by what each family is testing rather than by paper. Implementations are scoped to the local batch and task contract; see [`analysis/implementation_scope.md`](analysis/implementation_scope.md) for what each scoped version does and does not include.

| Family | Methods | What it is testing |
| --- | --- | --- |
| Reference baselines | `CE`, `REINFORCE`, `PG`, `TrajPG` | Sanity floor: where does supervised CE or vanilla PG land? |
| RLVR baselines | `GRPO`, `DrGRPO`, `DAPOLite` | Group-relative rewards, clipped surrogates, DAPO-style filtering and aggregation. |
| Candidate-target | `TPO`, `TPONoAnchor`, `GroupPG`, `TPOFullAction`, `TPOToken`, `GRPOToken` | Local candidate-simplex objectives at the sequence, full-action, or per-token level. |
| Influence (delight) | `DG`, `Kondo`, `DGToken` | The `advantage * surprisal` family, plus compute-aware screening and token-level credit. |
| Credit and geometry | `TEMPO`, `MaxRL`, `LogGrowth`, `PMDMean` | Prefix-tree credit, grouped mean normalization, and alternate objective shapes. |
| Replay and freshness | `ReplayDG`, `FreshDG` | DG composed with a replay buffer and explicit age weighting. |
| Robustness diagnostics | `DGEntropyGuard`, `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG`, `R2VPO`, `ASPO` | Entropy-collapse guards and reward-noise filters under controlled stress tests. |
| Dense-correction toys | `SelfDistillDG`, `SCOPELite` | Oracle-label bridges from sparse reward to token-level supervision. |

## Tasks

- `mnist`: one-step contextual bandit, 10 actions.
- `token_reversal`: autoregressive reversal, fractional or binary reward.
- `masked_reversal`: reversal with reward only on a scored suffix.
- `chain_reversal`: reversal with an ordered checkpoint reward chain.
- `chain_arithmetic`: copy a prefix, emit a modular answer; final-checkpoint reward.
- `format_answer`: emit a format token then an answer token; final-checkpoint reward.
- `lm_bandit`: next-token prediction with any HuggingFace causal LM.

## Experiment matrix

The map I use when deciding what to run.

| Question | Methods | Task and config | What to look at |
| --- | --- | --- | --- |
| Does delight allocate influence better than uniform PG? | `PG`, `ASPO`, `R2VPO`, `DG` | `mnist`, `token_reversal` | `test_error`, `reward`, `ratio_mean`, `gate_mean` |
| Do RLVR baselines change the update geometry? | `GRPO`, `DrGRPO`, `DAPOLite`, `TPO` | `token_reversal --group_size 8 --inner_epochs 4` | `mixed_group_rate`, `ratio_mean`, `reverse_kl_mean`, `entropy` |
| Is freshness-aware replay worth the complexity? | `DG`, `ReplayDG`, `FreshDG` | `token_reversal --delay 4` with capacity 5 (fixed age) or 32 (stress) | `replay_age`, `replay_priority`, `batch_age`, `test_error` |
| Do conservative gates resist proxy rewards? | `DG`, `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG` | `token_reversal --reward_noise 0.2 --reward_noise_mode false_positive_rare_token` | `gate_mean`, `uncertainty_mean`, `test_error` |
| Does token-level credit help partial rewards? | `DGToken`, `TEMPO`, `TPOToken`, `GRPOToken` | `masked_reversal` | `test_error`, `test_error_unscored`, `mixed_group_rate` |
| Can sparse rewards bootstrap dense correction? | `SelfDistillDG`, `SCOPELite`, `CE`, `DG` | `chain_reversal`, `chain_arithmetic`, `format_answer` | `chain_reward`, `correction_loss`, `distill_loss`, `test_error` |
| Where does entropy collapse first? | `DG`, `ASPO`, `R2VPO`, `DGEntropyGuard` | any sequence task with `--entropy_diagnostics true` | `batch_delta_entropy`, `cov_prob_delta_logit`, `entropy_drop_*` |
