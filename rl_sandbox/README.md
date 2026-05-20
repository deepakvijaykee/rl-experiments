# rl_sandbox

The toy sandbox itself: tasks small enough that I can inspect the gradient on every step, and a registry of update rules to compare on them. Top-level context, install steps, headline findings, and the figures behind every claim live in the [repository README](../README.md). This page is the working menu of methods and tasks, plus the experiment matrix I sit with when deciding what to run next.

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

I group methods by the question each family is putting to the data rather than by paper of origin. Two methods from the same paper can disagree about the choice that decides the run, and two methods from very different lineages can agree on it. Grouping by question keeps this table closer to how I actually reach for one method over another. Implementations are scoped to the local batch and task contract; the per-method scope notes, including where a scoped version departs from the published method, live in [`analysis/implementation_scope.md`](analysis/implementation_scope.md).

| Family | Methods | What I use this family for |
| --- | --- | --- |
| Reference baselines | `CE`, `REINFORCE`, `PG`, `TrajPG` | The no-frills floor. Supervised CE on the same data and plain policy gradient without shaping are the curves every more elaborate method gets read against. |
| RLVR baselines | `GRPO`, `DrGRPO`, `DAPOLite` | The RLVR recipe in three pieces. The question I am after is which ingredient (group standardization or the clip) is the one carrying weight at fixed rollout cost, and what each strict-clipping variant actually changes. |
| Candidate-target | `TPO`, `TPONoAnchor`, `GroupPG`, `TPOFullAction`, `TPOToken`, `GRPOToken` | Methods that build the update target from the whole rollout group rather than only the sampled action. The shared question is whether the K-candidate construction actually extracts more per group, and whether the same benefit carries down to token-level credit. |
| Influence (delight) | `DG`, `Kondo`, `DGToken` | The `advantage * surprisal` weighting story. The question is whether this kind of weighting produces a structurally different update from advantage-only weighting, or whether it only rescales the same gradient direction. |
| Credit and geometry | `TEMPO`, `MaxRL`, `LogGrowth`, `PMDMean` | Alternative credit-assignment trees and policy-mirror geometries. On these tasks I want to know whether any of them behaves meaningfully differently from ordinary policy gradient, or whether each reduces to it in practice. |
| Replay and freshness | `ReplayDG`, `FreshDG` | The replay diagnostics. Replay buys variance reduction and pays in staleness; I use these two methods to find where on the sample-age axis the crossover sits, and how buffer composition shifts it. |
| Robustness diagnostics | `DGEntropyGuard`, `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG`, `R2VPO`, `ASPO` | Six different choices of uncertainty proxy on the same axis. The question they each answer differently is which proxies preserve true reward signal at a given noise rate, and which strip it away as collateral. No method without ground truth escapes that trade-off; the family is a tour of where different proxies land on the curve. |
| Dense-correction toys | `SelfDistillDG`, `SCOPELite` | The dense-correction question. These methods use an oracle reviser to bootstrap dense token-level supervision from a sparse trajectory-level reward, on chain-structured tasks where the reward signal lives only at the trajectory level. |

## Tasks

The tasks are deliberately small. Each one is built so that one axis of the method comparison becomes visible without the other variables changing alongside it.

- `mnist`: one-step contextual bandit over 10 actions. The simplest setting for comparing influence-allocation choices, with no horizon to confound the gradient direction.
- `token_reversal`: autoregressive reversal with either fractional or binary reward. The default task for the influence, noise, and entropy sweeps.
- `masked_reversal`: reversal with reward only on a scored suffix. Built to surface the partial-credit problem by giving the comparison both scored and unscored positions to read.
- `chain_reversal`: reversal with an ordered checkpoint reward chain. Built for the dense-correction experiments, where the question is whether a sparse trajectory reward can bootstrap dense token supervision.
- `chain_arithmetic`: copy a prefix and emit a modular answer, with a final-checkpoint reward. Useful for testing self-distillation on chained-reasoning structure.
- `format_answer`: emit a format token followed by an answer token, with a final-checkpoint reward. Adds a format-versus-answer axis on top of correctness, which is the axis instruction-following objectives actually care about.
- `lm_bandit`: next-token prediction with any HuggingFace causal LM. The bridge from toy methods to a real model under the same trainer.

## Experiment matrix

The map I work from when deciding what to run next. Each row pairs a research question with the methods that disagree about the answer, the task and configuration that surface the disagreement, and the logged metrics whose comparison resolves it. The matrix is not exhaustive; the seven below are the comparisons that in practice have given the cleanest signal per unit of compute.

| Question | Methods | Task and config | Metrics that resolve it |
| --- | --- | --- | --- |
| Does `advantage * surprisal` weighting change the update trajectory relative to uniform PG, or only rescale it? | `PG`, `ASPO`, `R2VPO`, `DG` | `mnist`, `token_reversal` | `test_error`, `reward`, `ratio_mean`, `gate_mean` |
| What does group standardization combined with PPO-style clipping do to the update geometry at fixed rollout cost? | `GRPO`, `DrGRPO`, `DAPOLite`, `TPO` | `token_reversal --group_size 8 --inner_epochs 4` | `mixed_group_rate`, `ratio_mean`, `reverse_kl_mean`, `entropy` |
| How does replay degrade as effective sample age grows past the speed at which the policy is updating? | `DG`, `ReplayDG`, `FreshDG` | `token_reversal --delay 4` with capacity 5 (fixed age) or 32 (stress) | `replay_age`, `replay_priority`, `batch_age`, `test_error` |
| Which uncertainty proxies preserve true reward signal at noise rate 0.2 without sacrificing learning speed? | `DG`, `UncertaintyDG`, `FilteredDG`, `RewardVarianceDG` | `token_reversal --reward_noise 0.2 --reward_noise_mode false_positive_rare_token` | `gate_mean`, `uncertainty_mean`, `test_error` |
| Does token-level credit improve learning at scored positions without damaging unscored ones? | `DGToken`, `TEMPO`, `TPOToken`, `GRPOToken` | `masked_reversal` | `test_error`, `test_error_unscored`, `mixed_group_rate` |
| Can an oracle reviser bridge sparse trajectory reward to dense token supervision on chain-structured tasks? | `SelfDistillDG`, `SCOPELite`, `CE`, `DG` | `chain_reversal`, `chain_arithmetic`, `format_answer` | `chain_reward`, `correction_loss`, `distill_loss`, `test_error` |
| Where does each method first lose enough entropy to stop exploring meaningfully? | `DG`, `ASPO`, `R2VPO`, `DGEntropyGuard` | any sequence task with `--entropy_diagnostics true` | `batch_delta_entropy`, `cov_prob_delta_logit`, `entropy_drop_*` |
