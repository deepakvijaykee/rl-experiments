# Delightful Policy Gradient

PyTorch implementation of [Delightful Policy Gradient](https://arxiv.org/abs/2603.14608) (Osband 2026) with field baselines for comparison.

## Intuition

Standard policy gradients weight each sample by advantage alone. A rare wrong action gets the same gradient weight as a rare right one. DG gates each term with *delight* = advantage x surprisal. Rare successes (breakthroughs) pass through; rare failures (blunders) are suppressed. One sigmoid, one multiply, drop-in replacement for REINFORCE.

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
