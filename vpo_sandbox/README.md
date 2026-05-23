# vpo_sandbox

A toy Vector Policy Optimization sandbox for vector-reward, test-time-search experiments.

VPO does not fit the scalar `rl_sandbox.Batch` contract, and forcing it to would hide the part that makes VPO interesting. A rollout here is a *set* of candidate answers rather than a single answer. Each candidate carries a reward *vector* rather than a scalar. The training reward is computed at the set level, not the candidate level. Keeping VPO in its own package keeps that invariant visible instead of burying it behind optional fields in the scalar sandbox.

## What the local implementation keeps

Every prompt samples a GRPO group of rollout sets. Each rollout set contains `num_candidates` sampled answers, and each candidate has a vector reward `r(x, y)`. VPO draws a batch of Dirichlet scalarization weights and shares the same draws across every rollout in a prompt group, which is what makes within-group advantages comparable across rollout sets rather than across independently scalarized samples. The set reward is `mean_w max_y w^T r(x, y)`. That set reward is then group-normalized and applied through a PPO/GRPO clipped surrogate, with the same advantage broadcast to every candidate inside the set. `ScalarGRPO` and `MultiRLVR` are kept alongside as scalar baselines, with explicit config checks for single-answer versus multi-answer rollouts so that a misconfigured run fails loudly rather than silently dropping the candidate dimension.

The toy scope is deliberate. The policy is a prompt/slot embedding model rather than a causal LM, and the task is a two-objective Pareto-front bandit. That choice pins the reward estimator and the search-diversity contract without dragging in distributed rollout infrastructure, which is the part of the original VPO setup the sandbox is not trying to reproduce.

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

Scalar single-answer GRPO baseline:

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

The evaluation metrics that decide whether VPO is doing what it is supposed to do are `best_at_1`, `best_at_3`, `best_at_9`, and `pool_diversity_l1`. `best_at_k` uses the fixed scalar deployment reward and answers whether the candidate pool covers good answers under search. `pool_diversity_l1` measures spread in reward-vector space inside each emitted candidate set and answers whether the pool actually covers different trade-offs rather than collapsing to near-duplicates.
