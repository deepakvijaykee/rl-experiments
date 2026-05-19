# rl-experiments

A small PyTorch sandbox for analyzing RL update rules in toy settings. A second flow, [`rlm_grpo/`](rlm_grpo/), trains small causal LMs with GRPO and recursive, tree-of-rollouts sampling.

The OPD appendix lives separately in [`opd_sandbox/`](opd_sandbox/) so
distillation-specific experiments do not blur the reward-update sandbox.

The practical question: when feedback is sparse, noisy, or delayed, which samples should receive gradient credit?

This is not a full RLHF or RLVR system, and that is the point. A benchmark tells you which update rule got the right answer fastest. The sandbox is built to answer a different question: what does the rule *do* to the policy on the way there? Where does gradient mass concentrate? When does the importance ratio drift far enough that the surrogate stops tracking the objective? At what step does entropy collapse become irreversible? Those are the signals that distributed rollout systems amortize away, and this setup is built to keep them visible.

## How I think about the problem

Most post-training RL papers organize themselves by objective family: PPO, GRPO, DPO, RLVR, rejection sampling, distillation. I find that framing too coarse for picking what to try next. The more useful question, for me, sits one level below the objective:

> Given a fixed budget of online rollouts, learner updates, and KL distance from the reference policy, which samples should receive gradient weight, and at what granularity?

It breaks into five axes, each of which is a failure mode the sandbox can reproduce cheaply.

1. **Influence allocation.** Should every sample contribute uniformly, or should the update weight by advantage, surprisal, or some product of the two? `DG`-style delight (`advantage * surprisal`) is one bet. GRPO's group-normalized advantage is another. Plain REINFORCE is the null hypothesis.
2. **Credit granularity.** Trajectory, branch, or token. Sparse-reward tasks need a credit story even when the feedback is one bit. Dense-reward tasks waste signal if you flatten everything back to a sequence score. `TPOToken` and `DGToken` are the per-token bets here.
3. **Support and coverage.** When is logged data still informative? Old rollouts have stale importance ratios. Bigger replay buffers reduce variance, but only inside a freshness window. Outside that window you are training against a different policy. `ReplayDG` and `FreshDG` are the diagnostics.
4. **Reward uncertainty.** When does a rare high reward signal a breakthrough versus a proxy mistake? Filtering, uncertainty heuristics, and reward-variance penalties all try to answer this. They fail differently: wrong proxy granularity, premature entropy collapse, or just too slow. `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG`, `ASPO`, and `R2VPO` cover the spectrum from conservative gating to ratio-variance regularization.
5. **Optimization geometry.** Clipping, ratio variance, normalization, KL terms, entropy. The wrong combination collapses entropy inside 100 steps even when the loss curve looks healthy. The entropy-collapse sweep is the cheapest way to see this.

Each method in the sandbox makes a different bet on one or two of these axes. The repo lets me compare them on tasks small enough to inspect the gradient directly.

## What is in here

- [`rl_sandbox/`](rl_sandbox/) is the toy sandbox: bandit and sequence tasks, plus a method registry covering PG, GRPO, DG, TPO, and the smaller families that test specific axes (entropy guards, reward-noise filters, token credit, dense correction). See its [README](rl_sandbox/README.md) for the full method and task menu.
- [`opd_sandbox/`](opd_sandbox/) is the OPD appendix sandbox: exact reverse-KL and sampled-token OPD on student-sampled toy prefixes with a smoothed oracle teacher.
- [`rlm_grpo/`](rlm_grpo/) is the LM-scale flow. See its [README](rlm_grpo/README.md) for the training contract (root and child rollouts, reward propagation, child-count normalization) and CLI options.
- [`rl_sandbox/analysis/`](rl_sandbox/analysis/) is the evidence: reproduction commands, result tables, and figures from compact three-seed runs.

## Run this first

```bash
pip install -e .

python -m rl_sandbox.train --task token_reversal --method TPO \
  --batch_size 96 --group_size 8 --inner_epochs 4 \
  --num_steps 300 --eval_every 20 --num_seeds 3
```

The base install covers everything in this section. Add `[lm-bandit]` for the HuggingFace LM bandit task, or `[rlm]` for the [`rlm_grpo/`](rlm_grpo/) flow.

Then regenerate the figures:

```bash
python rl_sandbox/analysis/plot_evidence.py
```

The reward-noise, replay, partial-credit, dense-correction, and entropy sweeps live in [`rl_sandbox/analysis/sweep_manifest.md`](rl_sandbox/analysis/sweep_manifest.md). They run end to end on one GPU.

## What I take from these runs

I run three seeds per cell, small batches, short horizons. I treat these as regime checks. The absolute numbers shift between seeds; the ordering between methods and the way each one fails does not. Five inferences I would lead with, then the unpacked version of each.

1. `TPO` wins on clean token-reversal by treating the grouped rollouts as a soft target distribution; `GRPO`'s use of the same rollouts for advantage normalization alone leaves the candidate-target signal on the floor.
2. Replay's useful band is controlled by effective sample age, not buffer capacity. Any replay comparison that does not report an age distribution is reporting the wrong axis.
3. Token-level credit routes existing reward signal precisely but cannot create signal where none exists. The common trap is reaching for credit assignment when the problem is reward sparsity; the right tool there is reward shaping.
4. Every noise-robust method I tested trades exploration for noise rejection. The right framing is a calibration question against your specific noise rate; "pick a robust method" skips the trade-off the method is making.
5. `GRPO`'s entropy collapse is a normalization problem dressed up as a clipping problem. The fix lives upstream of the clip, which is why recent variants such as `DrGRPO` and `DAPO` get their traction by attacking the standardization rather than tightening the clip.

- `TPO` was the strongest method on clean token-reversal: faster than `GRPO` and `DG`, with final entropy near `DG`'s. The reason, as I read it, comes down to what each method does with the same grouped rollouts. `GRPO` uses them to compute an advantage, then applies that advantage to the one sampled action. `TPO` uses the same rollouts to build a soft target distribution across all the candidates and pulls the policy toward that target. Functionally that is the difference between a single-sample policy gradient and behavior cloning on a soft label, and the second carries strictly more information per step. The implication is unflattering for the standard GRPO recipe: if you already pay for group rollouts, you are leaving information on the floor by using them only to normalize advantage. Whether the gap survives at scale, with stronger base policies and richer rewards, I have not tested. The toy-task evidence is suggestive enough that I would reach for `TPO` before `GRPO` in a new sandbox.

- Replay is a Goldilocks problem. Capacity 5 at delay 4 is fine; capacity 32 collapses entropy unless freshness decay is strong. The familiar story is variance versus staleness: replay reduces gradient variance by recycling samples, but the recycled samples come from old policies. As the buffer grows, the average sample age grows with it, and past a point the gradient is pointing at where the policy *used to be*. What I take from this is that buffer capacity is the wrong unit. The variable that controls whether replay helps is effective sample age, which depends on buffer size, freshness decay, and how fast the policy is moving. Capacity is a coarse proxy and a misleading one once decay is in the picture. So when I see a replay-augmented method that reports buffer size without reporting the age distribution it produces, I now treat that as a missing diagnostic. The age distribution is the variable the method is tuning over, and it is what I would want to see before trusting any comparison.

- `TPOToken` drove the scored suffix of `masked_reversal` to zero error while unscored positions stayed near chance. Mechanically this is unsurprising: token-level credit routes existing reward signal to where it belongs, and it cannot create signal where there is none. The inference I almost missed is the one that matters. People reach for "smarter credit assignment" when the problem is sparse reward signal. These are different tools doing different jobs. Credit assignment is routing; reward shaping is adding signal. Conflating them produces a specific failure mode I almost walked into: instrument per-token credit on a task with sparse reward, then act surprised that nothing learns on the positions where there is no reward to route. The takeaway I would give a teammate is to ask first whether the reward function carries signal at the positions where you want learning. If it does not, the right move is reward design, not a fancier credit-assignment scheme.

- Reward-noise heuristics are regularization in disguise. `FilteredDG` failed in a boring way: its uncertainty proxy was batch-level in ungrouped runs, so the threshold either kept every batch or dropped every batch. `ASPO` was sturdier under false-positive rare-token noise, but at the cost of very low final entropy. The pattern is hard to escape. Every noise-robust method I tested makes the update less sensitive to the advantage when the reward is uncertain. That helps when the reward is wrong, and it also slows you down when the reward is right. There is no free filtering. What this implies is that "pick a robust method" is the wrong framing. The right framing is a calibration question: how much true signal am I willing to lose at my specific noise rate? Any robustness report should pair the noise level with the exploration cost; without that pairing the comparison is incomplete.

- `GRPO` collapses entropy fastest in this sweep, and I think I see why. Group-normalized advantages standardize rewards within each rollout group. When the rollouts in a group are mostly similar (which is most of the time on an easy task), the standardization amplifies small per-rollout differences into large advantages. The policy concentrates on whichever rollout happened to win, exploration shrinks, and entropy is gone inside 300 steps. The non-obvious part is where any fix has to live. The PPO heritage in `GRPO` puts a clip on the policy ratio, but the clip sits downstream of the standardization, so it does not constrain the amplification at all. What I now think is that `GRPO`'s entropy-collapse pathology is a normalization problem dressed up as a clipping problem. That would predict why recent variants such as `DrGRPO` and `DAPO` get traction by attacking the normalization side (removing reward-std normalization, decoupling clipping ranges) rather than tightening the clip further. If I were going to fix `GRPO` for entropy collapse, the clip is not where I would start.

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
