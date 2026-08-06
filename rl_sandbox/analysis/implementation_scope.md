# Implementation scope

Every method in this sandbox is a scoped reimplementation of a published method, or else a plain reference baseline. Each one stays faithful in the update rule, the credit-assignment scheme, and the normalization choices, and each is trimmed of the distributed-system scaffolding a single-GPU toy setup has no use for. Some scoping decisions would quietly change what a method means, for instance running advantage normalization in a regime where the original paper assumes a critic. The trainer rejects those configs, because an arm that silently estimates something other than what its name says is worse to have in a comparison than to leave out.

Some rows below carry a validity regime in place of a scope note, and the two say different things. A scope note reports that a method has been simplified. A validity regime reports that the derivation behind the method depends on a property of the task, such as binary rewards or grouped rollouts. Run it outside that regime and it still produces a number, but the number has stopped estimating what the method was built to estimate, and nothing in the output shows that. Those rows name the regime for exactly that reason.

## Reference baselines

| Method | Scope here |
| --- | --- |
| `CE` | Cross-entropy against true labels, ignoring RL experience. A true oracle on MNIST, where it is the exact supervised objective. On sequence tasks it is a dense upper bound rather than a like-for-like baseline, because it supervises every position while the RL reward signal does not. |
| `REINFORCE` | No off-policy correction at all, with stale actions used as sampled. Present as the floor that shows what the correction terms are buying. |
| `PG` | Per-token importance-weighted policy gradient. Exact for one-step bandits. On sequence tasks the per-token ratio is the standard approximation rather than an exact trajectory-level correction. |
| `TrajPG` | Trajectory-level importance weighting using the full product ratio, which is the exact off-policy correction before capping. Practical only on short sequences where the variance of the product stays manageable, and it collapses to `PG` on one-step bandits. |

## RLVR family

| Method | Scope here |
| --- | --- |
| `GRPO` | Group-normalized rewards, PPO-style clipped token surrogate, optional reverse-KL penalty. Out of scope: a critic, a distributed rollout service, long-response infrastructure. |
| `DrGRPO` | Group-centered rewards without reward-std normalization. Fixed-length toy tasks remove most observable response-length bias, so this arm isolates the normalization change rather than the length effect. |
| `DAPOLite` | Group filtering, decoupled clipping, token-level aggregation. Overlong-response reward shaping is omitted because toy responses have fixed length. `DAPO` is accepted as an alias. |

## Candidate-target family

| Method | Scope here |
| --- | --- |
| `TPO` | Sampled-candidate target policy optimization over grouped rollouts, with old-policy anchors and z-scored rewards. |
| `TPONoAnchor` | The ablation that drops the old-policy anchor, leaving candidate weights proportional to $\exp(\text{skill}/\eta)$. Isolates how much of TPO's behavior comes from the anchor rather than from the candidate construction. |
| `GroupPG` | Scalar-weighted grouped policy gradient using TPO's skill signal. Keeps TPO's scoring and discards its soft-target construction, which is the arm that separates "better weights" from "different target". |
| `TPOFullAction` | Single-sample full-action classification TPO, scoped to clean on-policy MNIST with one optimizer epoch. |
| `TPOToken` / `GRPOToken` | Per-prefix candidate-simplex objectives for dense token-reward reversal. These consume a dedicated token-candidate batch rather than ordinary rollout batches. |

## Influence and gating

| Method | Scope here |
| --- | --- |
| `DG` | Delightful policy gradient, faithful to the published gate, meaning $\mathrm{sigmoid}(\text{advantage} \cdot \text{surprisal} / \eta)$ scaling the advantage. No scoping changes to the update itself. |
| `Kondo` | The compute-efficient variant, screening samples before the learner forward pass by setting the gate threshold to a batch quantile targeting `keep_ratio`, then sampling $\mathrm{Bernoulli}(\mathrm{sigmoid}((\text{delight} - \lambda)/\eta))$. Faithful to Algorithm 1. The compute saving is real but small here, since the toy learner is cheap, so the arm earns its place through the update semantics rather than the speedup. |
| `DGToken` | Per-token return-to-go credit, where token $t$ is credited by the fraction of remaining tokens the actor got right. Under `score_mask` the numerator counts only scored positions and the baseline is zeroed at unscored ones. This is a partial-reward credit benchmark rather than an oracle: in an autoregressive task the unscored prefix still causally conditions the scored suffix, so unscored positions may deserve indirect gradient that this scheme does not give them. Any reading of its unscored-position damage has to carry that caveat. |
| `ASPO` | Asymmetric importance weighting, with the ratio inverted on positive-advantage tokens and one-sided clipping on each side. Faithful to the published rule. |
| `R2VPO` | Ratio-variance penalty replacing hard clipping, with no ratio capping, since the penalty is what bounds large ratios. |
| `DGEntropyGuard` | Sampled-action probability guard for entropy-collapse diagnostics, downweighting positive-advantage updates to already-high-probability actions while mostly leaving rare successes alone. It stops short of full covariance control so that it stays downstream of the standardization, which is what makes it the control arm for the entropy result. |

## Credit trees and geometry

| Method | Scope here |
| --- | --- |
| `TEMPO` | Toy nonparametric prefix-tree credit within grouped rollouts, adding a branch-gated TD term to the GRPO baseline so that it reduces exactly to GRPO at non-branching tokens. Validity regime: grouped rollouts that keep a mixed reward signal. Once groups stop being mixed, the prefix values it depends on stop varying and the method has nothing left to add. |
| `MaxRL` | Per-group mean-reward normalization instead of std, which is what makes the gradient an unbiased estimate of the maximum-likelihood gradient. Validity regime: binary rewards with grouped rollouts. Outside it the ML connection breaks and $1/\mathrm{mean}$ stops being a principled weighting. |
| `LogGrowth` | Kelly-optimal policy gradient by inverse-propensity weighting on exact-match success. Validity regime: binary exact-match one-step bandits, which requires rewards in $\{0, 1\}$, success revealing the correct label, and an unshaped advantage. |
| `PMDMean` | Policy mirror descent with a mean-reward partition approximation, regressing the trajectory-level log-ratio toward context-centered reward. It uses the exact conditional expected reward where the task exposes it and falls back to the batch mean otherwise. |

## Replay, uncertainty, and dense correction

| Method | Scope here |
| --- | --- |
| `ReplayDG` / `FreshDG` | Replay composition with exponential age weighting. Useful as freshness diagnostics, though the buffer is in-process and small, so the implementation is not a production replay system. |
| `UncertaintyDG` / `FilteredDG` / `RewardVarianceDG` | Three insertion points for one uncertainty signal, which is hand-rolled from batch statistics rather than produced by a learned verifier-confidence model. The signal is the within-group reward standard deviation when groups exist and the whole-batch standard deviation otherwise, and that ungrouped fallback is what turns `FilteredDG` into a batch-level switch. The degeneration is a property of the scoping, not of the published method. |
| `SelfDistillDG` / `SCOPELite` | Oracle-label dense-correction toys for studying the bridge from sparse trajectory reward to token supervision. `SelfDistillDG` lets DG's token gate decide where sparse reward becomes dense cross-entropy, while `SCOPELite` recycles failed trajectories by applying dense CE on the suffix starting at the first wrong token. In both, the reviser is the task's own labels rather than a learned reviser or a process reward model, and that single assumption is what makes these results non-transferable as they stand. |
