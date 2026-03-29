# rl-experiments

Playground for exploring RL ideas, both from the literature and personal explorations, in toy settings that run on a single laptop GPU.

## The meta question

Standard PG may be optimizing the wrong local object. The real question is: **what is the right influence function over sampled trajectories and tokens, if the goal is fastest true improvement per unit of online budget and KL budget?**

Current methods each solve one piece — PPO constrains update size, GRPO removes the value model, DPO goes offline, coverage papers argue for online data, Q* argues for token-level credit. But none address influence allocation from first principles: which samples and tokens should get gradient budget at all, and how much.

### Three budgets

LLM RL has a scarce effective update budget. Past samples are computationally cheap to reuse but statistically not free — they become stale, go off-support, replay reward noise, and give poor token-level credit.

- **Rollout budget**: expensive fresh on-policy data
- **Update budget**: backward/optimizer/KL budget on existing data
- **Support budget**: how far current data still tells you something useful about the current policy

### Five separable problems

1. **Influence allocation** — which samples/tokens deserve gradient budget?
2. **Credit granularity** — where in the sequence should credit land?
3. **Support/coverage** — can the learner see enough of the right modes?
4. **Conservatism under uncertainty** — how to avoid rewarding proxy mistakes?
5. **Optimization geometry** — how do KL, clipping, normalization, staleness distort the update?

DG (delight = advantage x surprisal) shows that the default answer to #1 — weight by advantage alone — is wrong. The genuine research directions are where influence allocation interacts with the other four: token-level delight for credit assignment, uncertainty-aware delight for conservatism, coverage-aware allocation for support, and staleness-aware gating for geometry.

The strongest path is not "improve DG a bit" but to build a theory of influence allocation for online LLM RL.

## Experiments

- **[delightful_policy_gradient/](delightful_policy_gradient/)** — Implementation of [Delightful Policy Gradient](https://arxiv.org/abs/2603.14608) with field baselines (DAPO, PMDMean, etc.). Tests influence allocation in policy gradients across MNIST bandits, token reversal, and LM next-token prediction.
