# rl_sandbox

Policy gradient experiments exploring influence allocation: which samples and tokens deserve gradient budget, and by what rule?

Centered on [Delightful Policy Gradient](https://arxiv.org/abs/2603.14608) (Osband 2026), with field baselines testing alternative mechanisms: asymmetric IS (ASPO), ratio-variance regularization (R2VPO), prefix-tree credit assignment (TEMPO), and others.

## Usage

```bash
pip install -r requirements.txt

python -m rl_sandbox.train --method DG --delay 30
python -m rl_sandbox.train --task mnist --method TPOFullAction
python -m rl_sandbox.train --task token_reversal --method GRPO --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method TPO --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method TPOToken --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method DAPOLite --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method FreshDG --replay_capacity 32 --delay 4
python -m rl_sandbox.train --task chain_reversal --method SelfDistillDG
python -m rl_sandbox.train --task chain_arithmetic --method DG --seq_len 3 --vocab_size 5
python -m rl_sandbox.train --task format_answer --method DG --seq_len 3 --vocab_size 5
python -m rl_sandbox.train --task lm_bandit --method DG --model_name distilgpt2 --batch_size 16 --lr 5e-5
python -m rl_sandbox.train --sweep
python -m rl_sandbox.plot results.csv
```

## Methods

| Methods | Category | Scope |
| --- | --- | --- |
| CE, REINFORCE, PG, TrajPG | reference PG baselines | Direct sandbox implementations; sequence PG uses the logged-token approximation unless `TrajPG` is selected. |
| ASPO, R2VPO, PMDMean, LogGrowth | objective geometry baselines | Scoped toy implementations of the core update rule; unsupported reward regimes are rejected. |
| GRPO, DrGRPO, DAPOLite | RLVR baselines | Scoped implementations of group-relative rewards, clipping, normalization, and DAPO-lite design choices. Large-scale rollout infrastructure is out of scope. |
| TPO, TPONoAnchor, GroupPG | sampled-candidate TPO baselines | Faithful local versions over grouped sampled rollouts. |
| TPOFullAction | full-action TPO baseline | Scoped to clean, on-policy MNIST bandit runs with one optimizer epoch. |
| TPOToken, GRPOToken | token-candidate TPO/GRPO baselines | Per-prefix candidate-simplex implementations for dense token-reward reversal tasks. |
| DG, Kondo, DGToken | paper-centered influence methods | Direct sandbox implementations of delight gating, compute-aware screening, and token return-to-go credit. |
| TEMPO, MaxRL | credit/normalization baselines | Toy prefix-tree credit and binary grouped mean-reward normalization; invalid reward regimes are rejected. |
| ReplayDG, FreshDG | freshness explorations | DG composed with replay and explicit age weighting. |
| DGEntropyGuard, UncertaintyDG, FilteredDG, RewardVarianceDG | robustness explorations | Local diagnostics/heuristics for entropy collapse and reward-noise conservatism, not full paper systems. |
| SelfDistillDG, SCOPELite | dense-credit explorations | Oracle-label toy bridges from sparse reward to dense correction; not learned reviser or PRM implementations. |

## Tasks

- **mnist**: contextual bandit (10 actions)
- **token_reversal**: autoregressive sequence reversal (fractional or binary reward)
- **masked_reversal**: partial-reward variant (only scored suffix positions affect reward)
- **chain_reversal**: ordered checkpoint reward-chain variant
- **chain_arithmetic**: copied-prefix, modular-answer, final-checkpoint reward chain
- **format_answer**: format-token, answer-token, final-checkpoint reward chain
- **lm_bandit**: next-token prediction with any HuggingFace causal LM
