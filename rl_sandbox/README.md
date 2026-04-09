# rl_sandbox

Policy gradient experiments exploring influence allocation: which samples and tokens deserve gradient budget, and by what rule?

Centered on [Delightful Policy Gradient](https://arxiv.org/abs/2603.14608) (Osband 2026), with field baselines testing alternative mechanisms: asymmetric IS (ASPO), ratio-variance regularization (R2VPO), prefix-tree credit assignment (TEMPO), and others.

## Usage

```bash
pip install -r requirements.txt

python -m rl_sandbox.train --method DG --delay 30
python -m rl_sandbox.train --task lm_bandit --method DG --model_name distilgpt2 --batch_size 16 --lr 5e-5
python -m rl_sandbox.train --sweep
python -m rl_sandbox.plot results.csv
```

## Methods

CE, REINFORCE, PG, ASPO, DG, Kondo, DGToken, TEMPO, LogGrowth, MaxRL, R2VPO, PMDMean

## Tasks

- **mnist**: contextual bandit (10 actions)
- **token_reversal**: autoregressive sequence reversal (fractional or binary reward)
- **masked_reversal**: partial-reward variant (only scored suffix positions affect reward)
- **lm_bandit**: next-token prediction with any HuggingFace causal LM
