# rl-experiments

A single-machine research sandbox for studying **influence allocation in
post-training RL**: which sampled trajectories, branches, and tokens deserve
gradient budget under rollout, update, support, staleness, reward-noise, and
entropy-collapse constraints.

The repo is not a leaderboard or a full distributed RLHF/RLVR reproduction. It
isolates update rules and diagnostics in toy settings where the behavior is
cheap to inspect and compare.

## Thesis

Post-training RL is often framed around the objective family: PPO, GRPO, DPO,
RLVR, rejection sampling, or distillation. This repo frames the problem one
level lower:

> What is the right influence function over sampled trajectories and tokens,
> if the goal is fastest improvement per unit of online rollout, learner update,
> and KL/support budget?

That decomposes into five concrete questions:

1. **Influence allocation**: which samples should receive gradient budget?
2. **Credit granularity**: should credit land on trajectories, branches, or tokens?
3. **Support and coverage**: when is sampled data still informative?
4. **Reward uncertainty**: when does rare high reward mean a breakthrough versus a proxy mistake?
5. **Optimization geometry**: how do clipping, ratios, normalization, replay, and entropy interact?

DG-style delight, `advantage * surprisal`, is one influence signal. The sandbox
compares it against RLVR baselines, TPO-style candidate targets, replay
freshness, uncertainty heuristics, entropy diagnostics, and dense correction
from sparse rewards.

## What Is Included

- **Core sandbox**: [rl_sandbox/](rl_sandbox/)
- **Reproducible evidence commands**: [rl_sandbox/analysis/sweep_manifest.md](rl_sandbox/analysis/sweep_manifest.md)
- **Compact results matrix**: [rl_sandbox/analysis/results_matrix.md](rl_sandbox/analysis/results_matrix.md)

## Current Findings

These are compact three-seed GPU checks, not final benchmark claims.

| Axis | Current evidence |
| --- | --- |
| Clean token reversal | `TPO` is strongest in the compact run: `0.2399 +/- 0.0573` final error versus `DG` at `0.3345 +/- 0.0043` and `GRPO` at `0.3536 +/- 0.0106`. |
| Reward noise | `ASPO` is slightly best under false-positive rare-token noise, but with much lower entropy than DG-style methods. `FilteredDG` is brittle because its current uncertainty proxy is batch-level in ungrouped runs. |
| Replay freshness | Fixed-age replay, `delay=4` with `replay_capacity=5`, makes `FreshDG` slightly more stable than delayed `DG`. Capacity `32` is a stale-buffer stress test and can collapse entropy. |
| Partial token credit | `TPOToken` drives masked-reversal scored suffix error to `0.0000 +/- 0.0000` while unscored positions remain near chance, showing targeted credit rather than dense sequence learning. |
| Dense correction | `SelfDistillDG` and `SCOPELite` solve `chain_reversal` faster than CE at the 1500-step horizon; the 300-step exact-match result was under-budget. |
| Entropy collapse | `GRPO` collapses entropy earliest in the compact entropy sweep. `TPO` keeps entropy near DG while reaching the best error, so its win is not just faster entropy collapse. |

## Method Taxonomy

| Category | Methods | Scope |
| --- | --- | --- |
| Reference baselines | `CE`, `REINFORCE`, `PG`, `TrajPG` | Standard supervised or policy-gradient references for toy tasks. |
| Canonical RLVR baselines | `GRPO`, `DrGRPO`, `DAPOLite` | Scoped implementations of group-relative rewards, clipping, normalization, and DAPO-lite design choices. |
| Candidate-target baselines | `TPO`, `TPONoAnchor`, `GroupPG`, `TPOFullAction`, `TPOToken`, `GRPOToken` | Local TPO-style candidate-simplex objectives, including full-action MNIST and per-prefix token candidates. |
| Influence methods | `DG`, `Kondo`, `DGToken` | Delight gating, compute-aware screening, and token return-to-go credit. |
| Credit and normalization | `TEMPO`, `MaxRL`, `LogGrowth`, `PMDMean` | Prefix-tree credit, binary grouped normalization, and alternate objective geometry. |
| Replay and freshness | `ReplayDG`, `FreshDG` | DG composed with replay sampling and explicit age weighting. |
| Robustness diagnostics | `DGEntropyGuard`, `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG`, `R2VPO`, `ASPO` | Entropy-collapse and reward-noise stress tests. |
| Dense-correction toys | `SelfDistillDG`, `SCOPELite` | Oracle-label bridges from sparse reward to dense token correction. |

Paper-named methods are scoped to the local batch/task contract. Large-system
components such as distributed rollout workers, learned critics, verifier
serving, and production reward pipelines are intentionally out of scope.

## Run First

Install dependencies:

```bash
pip install -r requirements.txt
```

Influence baseline:

```bash
python -m rl_sandbox.train --task token_reversal --method TPO \
  --batch_size 96 --group_size 8 --inner_epochs 4 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

Replay freshness:

```bash
python -m rl_sandbox.train --task token_reversal --method FreshDG \
  --batch_size 96 --delay 4 --replay_capacity 5 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

Reward-noise robustness:

```bash
python -m rl_sandbox.train --task token_reversal --method UncertaintyDG \
  --batch_size 96 --reward_noise 0.2 \
  --reward_noise_mode false_positive_rare_token \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

Partial token credit:

```bash
python -m rl_sandbox.train --task masked_reversal --method TPOToken \
  --batch_size 96 --group_size 8 --inner_epochs 4 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

Dense correction:

```bash
python -m rl_sandbox.train --task chain_reversal --method SCOPELite \
  --batch_size 96 --num_steps 1500 --eval_every 50 --num_seeds 3
```

Entropy diagnostics:

```bash
python -m rl_sandbox.train --task token_reversal --method DGEntropyGuard \
  --batch_size 96 --entropy_diagnostics true \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

For the full compact evidence suite, use
[rl_sandbox/analysis/sweep_manifest.md](rl_sandbox/analysis/sweep_manifest.md).

## Tasks

- `mnist`: one-step contextual bandit
- `token_reversal`: autoregressive sequence reversal
- `masked_reversal`: suffix-scored partial-credit reversal
- `chain_reversal`: ordered checkpoint reward-chain reversal
- `chain_arithmetic`: copy-plus-modular-answer reward chain
- `format_answer`: format-token plus answer-token reward chain
- `lm_bandit`: next-token LM bandit

## Verification

```bash
python -m compileall -q rl_sandbox
python -m rl_sandbox.train --task token_reversal --method DG \
  --batch_size 16 --num_steps 2 --eval_every 1 --num_seeds 1 \
  --output /tmp/rl_sandbox_smoke.csv --verbose false
```
