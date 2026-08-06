# vpo_sandbox

A toy Vector Policy Optimization sandbox for vector-reward, test-time-search experiments.

VPO does not fit the scalar `rl_sandbox.Batch` contract, and forcing it to would hide the part that makes VPO interesting. A rollout here is a set of candidate answers, each candidate carries a reward vector, and the training reward is computed over the whole set. Every one of those three departs from the scalar contract, and expressing them as optional fields on that contract would leave the reader unable to see which invariants a VPO run depends on. Its own package keeps them in the open.

## Why the set reward has this shape

The starting assumption is a deployment one. If test-time search is going to choose among several candidates, then training should be optimizing the quality of the pool the search draws from, not the quality of any single answer in it. The set reward is what encodes that:

```math
\text{set reward} = \mathbb{E}_{w \sim \mathrm{Dirichlet}}\bigl[\max_y w^\top r(x, y)\bigr]
```

The order of the two operators is the whole design. The max sits inside and the average sits outside, so the score asks whether the set contains a good answer for each scalarization drawn, rather than whether the set is good on average under the average scalarization. Swapping them would give $\max_y (\mathbb{E}_w[w])^\top r$, which is just best-of-n under the mean weight vector and carries no preference at all for how the candidates differ from one another. As written, the score is a maximum of linear functions of the reward vectors and therefore convex in the candidate set, which is what makes coverage pay: two candidates that each win on a different region of weight space score strictly higher than two copies of the compromise answer that wins on neither. This is where the pressure toward keeping a spread of trade-offs comes from, and it comes from the estimator rather than from an added diversity bonus.

Sharing the Dirichlet draws across every rollout set in a prompt group matters for a separate reason. Group advantages are computed by centering within the group, so if each rollout set were scored under its own independent weight draws, the differences between rollout sets would mix set quality with the noise of the weight sampling. Reusing one batch of draws across the group is the common-random-numbers trick: the comparison becomes paired, and what survives the centering is the difference between candidate sets rather than the difference between the scalarizations they happened to be judged under.

The set advantage is then broadcast unchanged to every candidate inside the set, which is a choice about credit granularity and worth understanding as one. The reward is a property of the set, so any candidate-level attribution would have to be invented rather than derived, and the uniform broadcast is what the reward actually licenses. A set that covered the weight space well reinforces every one of its members, including the candidate that lost under most draws precisely because it was off covering a corner.

## What the local implementation keeps

Every prompt samples a GRPO group of rollout sets. Each rollout set contains `num_candidates` sampled answers, and each candidate has a vector reward $r(x, y)$. VPO draws a batch of Dirichlet scalarization weights and shares those draws across every rollout in a prompt group, which is what makes within-group advantages comparable across rollout sets rather than across independently scalarized samples. The set reward $\mathbb{E}_w[\max_y w^\top r(x, y)]$ is then group-normalized and applied through a PPO/GRPO clipped surrogate, with the same advantage broadcast to every candidate inside the set.

`ScalarGRPO` and `MultiRLVR` are kept alongside as scalar baselines, and they bracket VPO from two directions. `ScalarGRPO` scores a single answer under the fixed deployment weights, so it is the ordinary single-answer case. `MultiRLVR` keeps the multi-candidate rollout but scores it as $\max_y w_\text{gold}^\top r$, best-of-n under one fixed scalarization, which isolates exactly the contribution of averaging over sampled weights: it gets the same rollout budget and the same max, and differs only in never seeing more than one direction in reward space. Both baselines carry explicit config checks for single-answer versus multi-answer rollouts, so a misconfigured run fails loudly rather than silently dropping the candidate dimension.

The scope stays small so that the two things under study stay visible. The policy is a prompt and answer-slot embedding model instead of a causal LM, and the task is a two-objective Pareto-front bandit. Between them they pin down the reward estimator and the search-diversity contract, and they leave out the distributed rollout infrastructure, which is the part of the original VPO setup this sandbox makes no attempt to reproduce.

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

Taken separately, each one is gameable. A policy can hold high diversity by emitting candidates that are merely different and uniformly mediocre, and it can hold a strong `best_at_1` by collapsing onto the single scalarized optimum, which is the exact behavior VPO exists to avoid. The claim worth testing is the joint one. VPO should widen the gap between `best_at_9` and `best_at_1` relative to the scalar baselines while keeping `best_at_1` competitive, and only that combination distinguishes extra candidates that carry useful alternatives from extra candidates that pad the set.
