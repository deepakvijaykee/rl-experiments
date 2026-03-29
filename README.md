# rl-experiments

Playground for exploring RL ideas, both from the literature and personal explorations, in toy settings that run on a single laptop GPU.

## The meta question

**What is the right influence function over sampled trajectories and tokens, if the goal is fastest improvement per unit of online budget and KL budget?**

Current methods each address a piece of this — PPO constrains update size, GRPO removes the value model, DPO moves offline, coverage papers argue for online data, Q* targets token-level credit. Influence allocation — which samples and tokens should receive gradient budget, and how much — remains underexplored as a first-principles question.

### Three budgets

LLM RL operates under three distinct scarce budgets. Past samples are computationally cheap to reuse but statistically not free — they become stale, go off-support, replay reward noise, and provide poor token-level credit.

- **Rollout budget**: fresh on-policy data (generation-bound)
- **Update budget**: backward/optimizer/KL budget on existing data
- **Support budget**: how far current data remains informative about the current policy

### Five separable problems

1. **Influence allocation** — which samples/tokens deserve gradient budget?
2. **Credit granularity** — where in the sequence should credit land?
3. **Support/coverage** — can the learner see enough of the right modes?
4. **Conservatism under uncertainty** — how to avoid rewarding proxy mistakes?
5. **Optimization geometry** — how do KL, clipping, normalization, staleness distort the update?

DG (delight = advantage x surprisal) addresses #1 by gating gradient terms with the interaction of reward signal and action probability. It connects to the Kelly criterion (log-optimal budget allocation across contexts), mirror descent (asymmetric trust regions that distinguish breakthroughs from blunders), and variational inference (the sigmoid gate as the optimal latent variable in max-entropy EM for policy search).

The interesting directions are where influence allocation interacts with the other four: token-level delight for credit granularity, uncertainty-aware weighting for conservatism, coverage-aware allocation for support, and staleness-aware gating for optimization geometry.

## Experiments

- **[delightful_policy_gradient/](delightful_policy_gradient/)** — Implementation of [Delightful Policy Gradient](https://arxiv.org/abs/2603.14608) with field baselines (DAPO, PMDMean, etc.). Tests influence allocation in policy gradients across MNIST bandits, token reversal, and LM next-token prediction.
