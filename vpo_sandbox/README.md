# VPO Sandbox

A toy Vector Policy Optimization sandbox for vector-reward, test-time-search
experiments.

VPO does not fit the scalar `rl_sandbox.Batch` contract. A rollout here is a set
of candidate answers, each candidate receives a reward vector, and the training
reward is computed at the set level. Keeping this separate makes the VPO
invariant visible instead of hiding it behind optional fields in the scalar
sandbox.

## What Is Faithful Here

- Each prompt samples a GRPO group of rollout sets.
- Each rollout set contains `num_candidates` sampled answers.
- Each candidate has a vector reward `r(x, y)`.
- VPO samples Dirichlet scalarization weights and shares the same weight draws
  across every rollout in a prompt group.
- The set reward is `mean_w max_y w^T r(x, y)`.
- The set reward is group-normalized and applied through a PPO/GRPO clipped
  surrogate, with the same advantage broadcast to every candidate in the set.
- `ScalarGRPO` and `MultiRLVR` are kept as scalar baselines with explicit config
  checks for single-answer versus multi-answer rollouts.

What is intentionally toy-scoped: the policy is a prompt/slot embedding model,
not a causal LM, and the task is a two-objective Pareto-front bandit. This pins
the reward estimator and search-diversity contract without importing distributed
rollout infrastructure.

## Run

```bash
python -m vpo_sandbox.train \
  --method VPOGRPO \
  --batch_size 128 \
  --group_size 8 \
  --num_candidates 3 \
  --num_steps 500 \
  --eval_every 20 \
  --num_seeds 3
```

Scalar single-answer GRPO:

```bash
python -m vpo_sandbox.train \
  --method ScalarGRPO \
  --num_candidates 1 \
  --batch_size 128 \
  --group_size 8
```

Multi-answer scalar baseline:

```bash
python -m vpo_sandbox.train \
  --method MultiRLVR \
  --num_candidates 3 \
  --batch_size 128 \
  --group_size 8
```

The main evaluation metrics are `best_at_1`, `best_at_3`, `best_at_9`, and
`pool_diversity_l1`. Best-at-k uses the fixed scalar deployment reward, while
`pool_diversity_l1` measures spread in reward-vector space inside each emitted
candidate set.

