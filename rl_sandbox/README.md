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

Core baselines:

CE, REINFORCE, PG, TrajPG, ASPO, GRPO, DrGRPO, DAPOLite, TPO,
TPONoAnchor, GroupPG, TPOFullAction, TPOToken, GRPOToken, DG, Kondo,
DGToken, TEMPO, LogGrowth, MaxRL, R2VPO, PMDMean

Research-axis explorations:

ReplayDG, FreshDG, DGEntropyGuard, UncertaintyDG, FilteredDG,
RewardVarianceDG, SelfDistillDG, SCOPELite

See [../docs/implementation_fidelity.md](../docs/implementation_fidelity.md)
for the exact scope of each paper-named method.

## Tasks

- **mnist**: contextual bandit (10 actions)
- **token_reversal**: autoregressive sequence reversal (fractional or binary reward)
- **masked_reversal**: partial-reward variant (only scored suffix positions affect reward)
- **chain_reversal**: ordered checkpoint reward-chain variant
- **chain_arithmetic**: copied-prefix, modular-answer, final-checkpoint reward chain
- **format_answer**: format-token, answer-token, final-checkpoint reward chain
- **lm_bandit**: next-token prediction with any HuggingFace causal LM
