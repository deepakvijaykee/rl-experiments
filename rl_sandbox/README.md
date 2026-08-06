# rl_sandbox

The toy sandbox itself. The tasks are small enough that I can inspect the gradient on every step, and the registry holds the update rules I want to compare on them. Top-level context, install steps, headline findings, and the figures behind every claim live in the [repository README](../README.md). This page is the working menu of methods and tasks, plus the experiment matrix I sit with when deciding what to run next.

## Quick runs

```bash
pip install -e .              # base sandbox
pip install -e ".[lm-bandit]" # add for the lm_bandit task

python -m rl_sandbox.train --method DG --delay 30
python -m rl_sandbox.train --task mnist --method TPOFullAction
python -m rl_sandbox.train --task token_reversal --method GRPO --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method TPO --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method TPOToken --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method DAPOLite --group_size 8 --inner_epochs 4
python -m rl_sandbox.train --task token_reversal --method FreshDG --delay 4 --replay_capacity 5
python -m rl_sandbox.train --task chain_reversal --method SelfDistillDG
python -m rl_sandbox.train --task chain_arithmetic --method DG --seq_len 3 --vocab_size 5
python -m rl_sandbox.train --task format_answer --method DG --seq_len 3 --vocab_size 5
python -m rl_sandbox.train --task lm_bandit --method DG --model_name distilgpt2 --batch_size 16 --lr 5e-5
python -m rl_sandbox.train --sweep
python -m rl_sandbox.plot results.csv
```

Reproduction commands and result tables live in [`analysis/sweep_manifest.md`](analysis/sweep_manifest.md) and [`analysis/results_matrix.md`](analysis/results_matrix.md).

## Methods

I group methods by the question each family is putting to the data rather than by paper of origin. Two methods from the same paper can disagree about the choice that decides the run, and two methods from very different lineages can agree on it, so grouping by question keeps this table closer to how I actually reach for one method over another. Every implementation is scoped to the local batch and task contract, and the per-method scope notes, including where a scoped version departs from the published method, live in [`analysis/implementation_scope.md`](analysis/implementation_scope.md).

| Family | Methods | What I use this family for |
| --- | --- | --- |
| Reference baselines | `CE`, `REINFORCE`, `PG`, `TrajPG` | The no-frills floor. Supervised CE on the same data and plain policy gradient without shaping are the curves every more elaborate method gets read against. |
| RLVR baselines | `GRPO`, `DrGRPO`, `DAPOLite` | The RLVR recipe in three pieces. What I am after is which ingredient, group standardization or the clip, carries the weight at fixed rollout cost, and what each strict-clipping variant actually changes. `DAPO` is accepted as an alias for `DAPOLite`. |
| Candidate-target | `TPO`, `TPONoAnchor`, `GroupPG`, `TPOFullAction`, `TPOToken`, `GRPOToken` | Methods that build the update target from the whole rollout group rather than only the sampled action. The shared question is whether the K-candidate construction really extracts more per group, and whether the same benefit carries down to token-level credit. |
| Influence (delight) | `DG`, `Kondo`, `DGToken` | Delightful Policy Gradient ([arXiv:2603.14608](https://arxiv.org/abs/2603.14608)) and its compute-efficient variant Kondo ([arXiv:2603.20526](https://arxiv.org/abs/2603.20526)). The name is misleading if read as `advantage * surprisal` weighting: that product is the argument to a sigmoid gate, and the gate multiplies the advantage, so surprising successes get amplified and surprising failures get suppressed. Kondo moves the same gate before the learner forward pass and samples it stochastically, so screening saves compute rather than only reweighting. |
| Credit and geometry | `TEMPO`, `MaxRL`, `LogGrowth`, `PMDMean` | Alternative credit-assignment trees and policy-mirror geometries. TEMPO ([arXiv:2509.18314](https://arxiv.org/abs/2509.18314)) adds a branch-gated TD correction on top of the GRPO baseline, so it reduces exactly to GRPO wherever rollouts have not yet diverged. MaxRL ([arXiv:2602.02710](https://arxiv.org/abs/2602.02710)) normalizes by group mean rather than std and is only meaningful for binary rewards with grouped rollouts. PMDMean follows the Kimi k1.5 lineage ([arXiv:2501.12599](https://arxiv.org/abs/2501.12599)). What I want from all four is whether any behaves meaningfully differently from ordinary policy gradient on these tasks, or whether each reduces to it in practice. |
| Replay and freshness | `ReplayDG`, `FreshDG` | The replay diagnostics. Replay buys variance reduction and pays in staleness, so I use these two to find where on the sample-age axis the crossover sits and how buffer composition shifts it. |
| Uncertainty insertion points | `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG` | The cleanest three-way comparison in the sandbox, because all three consume the same uncertainty signal, the within-group reward standard deviation, and differ only in where it enters. UncertaintyDG subtracts it inside the gate argument, RewardVarianceDG divides the advantage by it before the gate sees it, and FilteredDG applies a hard keep-or-drop mask outside the gate. Holding the signal fixed is what makes the resulting failure modes attributable to the insertion point rather than to three different definitions of uncertainty. |
| Ratio handling under noise | `ASPO`, `R2VPO`, `DGEntropyGuard` | The same axis approached through the importance ratio. ASPO ([arXiv:2510.06062](https://arxiv.org/abs/2510.06062)) inverts the ratio on positive-advantage tokens so rare good actions get more gradient rather than less, with one-sided clipping to bound it. R2VPO ([arXiv:2601.03320](https://arxiv.org/abs/2601.03320)) removes hard clipping in favor of a ratio-variance penalty, which preserves a rare high-advantage sample instead of zeroing it at a clip boundary. Since DG and ASPO amplify rare breakthroughs while R2VPO only declines to destroy them, the pair asks whether amplification is necessary or preservation suffices. `DGEntropyGuard` sits here as the downstream control. |
| Dense-correction toys | `SelfDistillDG`, `SCOPELite` | The dense-correction question. These methods use an oracle reviser to bootstrap dense token-level supervision from a sparse trajectory-level reward, on chain-structured tasks where the reward signal lives only at the trajectory level. |

## Tasks

The tasks are deliberately small. Each one is built so that a single axis of the method comparison becomes visible without the other variables moving alongside it.

- `mnist` is a one-step contextual bandit over 10 actions, and it is the simplest setting for comparing influence-allocation choices because there is no horizon to confound the gradient direction.
- `token_reversal` is autoregressive reversal with either fractional or binary reward. It is the default task for the influence, noise, and entropy sweeps.
- `masked_reversal` is reversal with reward only on a scored suffix, built to surface the partial-credit problem by giving the comparison both scored and unscored positions to read.
- `chain_reversal` is reversal with an ordered checkpoint reward chain, built for the dense-correction experiments, where the question is whether a sparse trajectory reward can bootstrap dense token supervision.
- `chain_arithmetic` asks the policy to copy a prefix and emit a modular answer under a final-checkpoint reward, which makes it useful for testing self-distillation on chained-reasoning structure.
- `format_answer` asks for a format token followed by an answer token, again under a final-checkpoint reward. It adds a format-versus-answer axis on top of correctness, which is the axis instruction-following objectives actually care about.
- `lm_bandit` is next-token prediction with any HuggingFace causal LM, and it is the bridge from toy methods to a real model under the same trainer.

## Experiment matrix

The map I work from when deciding what to run next. Each row pairs a research question with the methods that disagree about the answer, the task and configuration that surface the disagreement, and the logged metrics whose comparison resolves it. The matrix is not exhaustive, and the rows below are simply the comparisons that have in practice given the cleanest signal per unit of compute.

These rows are not the seven axes from the [repository README](../README.md), and the two enumerations coinciding at seven is a coincidence worth not reading into. The axes are a taxonomy of design choices, two of which (privileged rollout acquisition and search coverage under vector rewards) change the batch contract and therefore live in the companion sandboxes rather than here. The matrix is a list of experiments, so it covers the five axes this sandbox can exercise and adds a sixth comparison, dense correction, which is a follow-up question rather than an axis of its own.

| Question | Methods | Task and config | Metrics that resolve it |
| --- | --- | --- | --- |
| Does DG's sigmoid gate change the update trajectory relative to uniform PG, or only rescale it per sample? | `PG`, `ASPO`, `R2VPO`, `DG` | `mnist`, `token_reversal` | `test_error`, `reward`, `ratio_mean`, `gate_mean` |
| What does group standardization combined with PPO-style clipping do to the update geometry at fixed rollout cost? | `GRPO`, `DrGRPO`, `DAPOLite`, `TPO` | `token_reversal --group_size 8 --inner_epochs 4` | `mixed_group_rate`, `ratio_mean`, `reverse_kl_mean`, `entropy` |
| How does replay degrade as effective sample age grows past the speed at which the policy is updating? | `DG`, `ReplayDG`, `FreshDG` | `token_reversal --delay 4` with capacity 5 (fixed age) or 32 (stress) | `replay_age`, `replay_priority`, `batch_age`, `test_error` |
| Which uncertainty proxies preserve true reward signal at noise rate 0.2 without sacrificing learning speed? | `DG`, `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG` | `token_reversal --reward_noise 0.2 --reward_noise_mode false_positive_rare_token` | `gate_mean`, `uncertainty_mean`, `test_error` |
| Does token-level credit improve learning at scored positions without damaging unscored ones? | `DGToken`, `TEMPO`, `TPOToken`, `GRPOToken` | `masked_reversal` | `test_error`, `test_error_unscored`, `mixed_group_rate` |
| Can an oracle reviser bridge sparse trajectory reward to dense token supervision on chain-structured tasks? | `SelfDistillDG`, `SCOPELite`, `CE`, `DG` | `chain_reversal`, `chain_arithmetic`, `format_answer` | `chain_reward`, `correction_loss`, `distill_loss`, `test_error` |
| Where does each method first lose enough entropy to stop exploring meaningfully? | `DG`, `ASPO`, `R2VPO`, `DGEntropyGuard` | any sequence task with `--entropy_diagnostics true` | `batch_delta_entropy`, `cov_prob_delta_logit`, `entropy_drop_*` |

The first row is worth answering partly on paper before spending compute on it, because half of it is settled by the form of the update. DG's gate is a positive scalar in `(0, 1)` multiplying the advantage, so on any single sample it can only rescale the policy-gradient direction and never rotate it. What it does not leave alone is the aggregate. The gate depends on the sample's own surprisal and is asymmetric in the sign of the advantage, so it changes the relative weight of samples within a batch, and the summed direction is not a scalar multiple of the plain policy-gradient direction. That is why `gate_mean` is logged next to `test_error`: a run where the gate sits flat near one half is one where DG has degenerated into scaled PG, and the comparison that step is meant to resolve has quietly stopped being a comparison.
