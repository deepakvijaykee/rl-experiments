# vpo_sandbox

A toy Vector Policy Optimization sandbox for vector-reward, test-time-search experiments.

VPO does not fit the scalar `rl_sandbox.Batch` contract, and forcing it to would hide the part that makes VPO interesting. A rollout here is a set of candidate answers rather than a single answer, each candidate carries a reward vector rather than a scalar, and the training reward is computed at the set level rather than the candidate level. Keeping VPO in its own package keeps those three invariants visible instead of burying them behind optional fields in the scalar sandbox.

## Why the set reward has this shape

The starting assumption is a deployment one. If test-time search is going to choose among several candidates, then training should be optimizing the quality of the pool the search draws from, not the quality of any single answer in it. The set reward is what encodes that:

```text
set reward = mean_w max_y w^T r(x, y),   w ~ Dirichlet
```

The order of the two operators is the whole design. The max sits inside and the average sits outside, so the score asks whether the set contains a good answer for each scalarization drawn, rather than whether the set is good on average under the average scalarization. Swapping them would give `max_y (mean_w w)^T r`, which is just best-of-n under the mean weight vector and carries no preference at all for how the candidates differ from one another. As written, the score is a maximum of linear functions of the reward vectors and therefore convex in the candidate set, which is what makes coverage pay: two candidates that each win on a different region of weight space score strictly higher than two copies of the compromise answer that wins on neither. This is where the pressure toward keeping a spread of trade-offs comes from, and it comes from the estimator rather than from an added diversity bonus.

Sharing the Dirichlet draws across every rollout set in a prompt group matters for a separate reason. Group advantages are computed by centering within the group, so if each rollout set were scored under its own independent weight draws, the differences between rollout sets would mix set quality with the noise of the weight sampling. Reusing one batch of draws across the group is the common-random-numbers trick: the comparison becomes paired, and what survives the centering is the difference between candidate sets rather than the difference between the scalarizations they happened to be judged under.

The set advantage is then broadcast unchanged to every candidate inside the set. That is a deliberate choice of credit granularity, not an oversight. The reward is a property of the set, so no candidate-level attribution is available without inventing one, and the uniform broadcast is the assumption-free option: a set that covered the weight space well reinforces all of its members, including the candidate that happened to lose under most draws but was the one covering the corner.

## What the local implementation keeps

Every prompt samples a GRPO group of rollout sets. Each rollout set contains `num_candidates` sampled answers, and each candidate has a vector reward `r(x, y)`. VPO draws a batch of Dirichlet scalarization weights and shares those draws across every rollout in a prompt group, which is what makes within-group advantages comparable across rollout sets rather than across independently scalarized samples. The set reward `mean_w max_y w^T r(x, y)` is then group-normalized and applied through a PPO/GRPO clipped surrogate, with the same advantage broadcast to every candidate inside the set.

`ScalarGRPO` and `MultiRLVR` are kept alongside as scalar baselines, and they bracket VPO from two directions. `ScalarGRPO` scores a single answer under the fixed deployment weights, so it is the ordinary single-answer case. `MultiRLVR` keeps the multi-candidate rollout but scores it as `max_y w_gold^T r`, best-of-n under one fixed scalarization, which isolates exactly the contribution of averaging over sampled weights: it gets the same rollout budget and the same max, and differs only in never seeing more than one direction in reward space. Both baselines carry explicit config checks for single-answer versus multi-answer rollouts, so a misconfigured run fails loudly rather than silently dropping the candidate dimension.

The toy scope is deliberate. The policy is a prompt and answer-slot embedding model rather than a causal LM, and the task is a two-objective Pareto-front bandit. That choice pins down the reward estimator and the search-diversity contract without dragging in distributed rollout infrastructure, which is the part of the original VPO setup the sandbox is not trying to reproduce.

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

## What to read off the run

The metrics that decide whether VPO is doing what it is supposed to do are `best_at_1`, `best_at_3`, `best_at_9`, and `pool_diversity_l1`, and they have to be read together because either one alone can be satisfied the wrong way. The `best_at_k` family uses the fixed scalar deployment reward and answers whether the candidate pool covers good answers under search. `pool_diversity_l1` measures spread in reward-vector space inside each emitted candidate set and answers whether the pool genuinely covers different trade-offs rather than collapsing to near-duplicates.

Taken separately they are both gameable. A policy can hold high diversity by emitting candidates that are merely different and uniformly mediocre, and it can hold a strong `best_at_1` by collapsing onto the single scalarized optimum, which is the behavior VPO exists to avoid. The claim worth testing is the joint one: VPO should widen the gap between `best_at_9` and `best_at_1` relative to the scalar baselines while keeping `best_at_1` competitive, since that combination is what says the extra candidates are carrying useful alternatives rather than padding.
