# rl-experiments

A small PyTorch sandbox for analyzing RL update rules in toy settings. A second flow, [`rlm_grpo/`](rlm_grpo/), trains 0.5B–0.6B Hugging Face causal LMs with GRPO and recursive, tree-of-rollouts sampling.

The practical question: when feedback is sparse, noisy, or delayed, which samples should actually receive gradient credit?

This is not a full RLHF or RLVR system, and that is the point. A benchmark tells you which update rule got the right answer fastest. The sandbox is built to answer a different question: what does the rule *do* to the policy on the way there? Where does gradient mass concentrate? When does the importance ratio drift far enough that the surrogate stops tracking the objective? At what step does entropy collapse become irreversible? Those are the signals that distributed rollout systems amortize away, and this setup is built to keep them visible.

## How I think about the problem

Most post-training RL papers organize themselves by objective family: PPO, GRPO, DPO, RLVR, rejection sampling, distillation. I find that framing too coarse for picking what to try next. The more useful question, for me, sits one level below the objective:

> Given a fixed budget of online rollouts, learner updates, and KL distance from the reference policy, which samples should receive gradient weight, and at what granularity?

It breaks into five axes, each of which is a failure mode the sandbox can reproduce cheaply.

1. **Influence allocation.** Should every sample contribute uniformly, or should the update weight by advantage, surprisal, or some product of the two? `DG`-style delight (`advantage × surprisal`) is one bet. GRPO's group-normalized advantage is another. Plain REINFORCE is the null hypothesis.
2. **Credit granularity.** Trajectory, branch, or token. Sparse-reward tasks need a credit story even when the feedback is one bit. Dense-reward tasks waste signal if you flatten everything back to a sequence score. `TPOToken` and `DGToken` are the per-token bets here.
3. **Support and coverage.** When is logged data still informative? Old rollouts have stale importance ratios. Bigger replay buffers reduce variance, but only inside a freshness window. Outside that window you are training against a different policy. `ReplayDG` and `FreshDG` are the diagnostics.
4. **Reward uncertainty.** When does a rare high reward signal a breakthrough versus a proxy mistake? Filtering, uncertainty heuristics, and reward-variance penalties all try to answer this. They fail differently: wrong proxy granularity, premature entropy collapse, or just too slow. `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG`, `ASPO`, and `R2VPO` cover the spectrum from conservative gating to ratio-variance regularization.
5. **Optimization geometry.** Clipping, ratio variance, normalization, KL terms, entropy. The wrong combination collapses entropy inside 100 steps even when the loss curve looks healthy. The entropy-collapse sweep is the cheapest way to see this.

Each method in the sandbox makes a different bet on one or two of these axes. The repo lets me compare them on tasks small enough to inspect the gradient directly.

## What is in here

- [`rl_sandbox/`](rl_sandbox/) is the toy sandbox: bandit and sequence tasks, plus a method registry covering PG, GRPO, DG, TPO, and the smaller families that test specific axes (entropy guards, reward-noise filters, token credit, dense correction). See its [README](rl_sandbox/README.md) for the full method and task menu.
- [`rlm_grpo/`](rlm_grpo/) is the LM-scale flow. See its [README](rlm_grpo/README.md) for the training contract (root and child rollouts, reward propagation, child-count normalization) and CLI options.
- [`rl_sandbox/analysis/`](rl_sandbox/analysis/) is the evidence: reproduction commands, result tables, and figures from compact three-seed runs.

## Run this first

```bash
pip install -r requirements.txt

python -m rl_sandbox.train --task token_reversal --method TPO \
  --batch_size 96 --group_size 8 --inner_epochs 4 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

Then regenerate the figures:

```bash
python rl_sandbox/analysis/plot_evidence.py
```

The reward-noise, replay, partial-credit, dense-correction, and entropy sweeps live in [`rl_sandbox/analysis/sweep_manifest.md`](rl_sandbox/analysis/sweep_manifest.md). They run end to end on one GPU.

## What I actually learned

Three seeds per cell, small batches, short horizons. The absolute numbers are vibes-level. Ordering between methods and the shape of the failure modes hold up across seeds, and that is what I trust.

- Sampled-candidate `TPO` was the strongest method on clean token-reversal. It learned faster than `GRPO` and `DG`, and it kept entropy at roughly `DG`'s level while doing so. That rules out the obvious story that its win is just faster collapse; the candidate-target update appears to be doing structural work beyond shrinking the action distribution.
- Replay only helps in a narrow band. A fixed-age stale buffer (capacity 5 at delay 4) is fine. A bigger buffer (capacity 32) is a stale-buffer trap: samples become much older than the nominal delay, and entropy collapses unless freshness decay is strong enough to suppress the old ones. The lesson generalizes: stale-data variance reduction is a Goldilocks problem.
- Token-level credit lands exactly where the reward does. `TPOToken` drives the scored suffix of `masked_reversal` to zero error while unscored positions stay near chance. Good for partial credit. Also a sharp reminder that the reward shape is doing most of the work upstream. Dense token reward is not a free lunch you can replace with smarter credit assignment.
- Reward-noise heuristics break for boring reasons. `FilteredDG` looked promising until I checked: its uncertainty proxy is batch-level in ungrouped runs, so the threshold either keeps every batch or drops every batch. `ASPO` is more robust under false-positive rare-token noise, but it pays for that with very low entropy — a different failure mode hiding inside a robustness win.
- `GRPO` collapses entropy earliest in the entropy sweep, even when its final accuracy is comparable to DG-class methods. The toy setup catches this inside 300 steps.

## Evidence plots

Mean and standard error across three seeds. Full trajectories rather than only final bars, so stalls and collapses are visible.

| Influence | Reward noise |
| --- | --- |
| ![Clean token-reversal learning curves](rl_sandbox/analysis/figures/influence.png) | ![False-positive reward-noise learning curves](rl_sandbox/analysis/figures/reward_noise.png) |

| Replay | Partial credit |
| --- | --- |
| ![Replay freshness trajectories under delay](rl_sandbox/analysis/figures/replay.png) | ![Masked-reversal scored and unscored trajectories](rl_sandbox/analysis/figures/partial_credit.png) |

| Dense correction | Entropy |
| --- | --- |
| ![Reward-chain dense correction trajectories](rl_sandbox/analysis/figures/dense_correction.png) | ![Entropy and accuracy trajectories](rl_sandbox/analysis/figures/entropy.png) |

Full numbers live in [`rl_sandbox/analysis/results_matrix.md`](rl_sandbox/analysis/results_matrix.md). Implementation-scope notes (what each method does and does not include here) are in [`rl_sandbox/analysis/implementation_scope.md`](rl_sandbox/analysis/implementation_scope.md).

## Scope

Single machine. CUDA when available, CPU otherwise. The sandbox does not include distributed rollout, a learned critic, or a production reward pipeline. Method implementations are scoped to the local batch and task contract: the goal is to inspect the update rule itself, on a setup small enough that the gradient and entropy signals are readable. Where scoping the method would silently change its meaning (for example, normalizing rewards in a regime where the paper assumes a critic), the trainer rejects the config rather than running a misleading variant.

## Verification

```bash
python -m compileall -q rl_sandbox
python -m rl_sandbox.train --task token_reversal --method DG \
  --batch_size 16 --num_steps 2 --eval_every 1 --num_seeds 1 \
  --output /tmp/rl_sandbox_smoke.csv --verbose false
```
