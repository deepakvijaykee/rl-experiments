# Implementation Scope

This sandbox isolates update rules and diagnostics that fit the local task and
batch contracts. It is not a full-scale reproduction suite for distributed
post-training systems.

| Method family | Scope in this repo |
| --- | --- |
| `GRPO` | Group-normalized rewards, PPO-style clipped token surrogate, and optional reverse-KL penalty. No critic, distributed rollout service, or long-response infrastructure. |
| `DrGRPO` | Group-centered rewards without reward-std normalization. Fixed-length tasks remove most observable response-length bias. |
| `DAPOLite` | Group filtering, decoupled clipping, and token-level aggregation. Overlong reward shaping is omitted because toy responses have fixed length. |
| `TPO` | Sampled-candidate target policy optimization over grouped rollouts using old-policy anchors and z-scored rewards. |
| `TPOFullAction` | Single-sample full-action classification TPO scoped to clean, on-policy MNIST with one optimizer epoch. |
| `TPOToken` / `GRPOToken` | Per-prefix candidate-simplex objectives for dense token-reward reversal tasks. They use a dedicated token-candidate batch, not ordinary rollout batches. |
| `TEMPO` | Toy nonparametric prefix-tree credit within grouped rollouts. It is valid only when groups keep mixed reward signal. |
| `ReplayDG` / `FreshDG` | Replay composition and explicit exponential age weighting. They are freshness diagnostics, not a production replay system. |
| `UncertaintyDG` / `FilteredDG` / `RewardVarianceDG` | Reward-disagreement heuristics for controlled proxy-noise stress tests. They are not learned verifier-confidence models. |
| `DGEntropyGuard` | Local sampled-action probability guard for entropy-collapse diagnostics, not a full covariance-control implementation. |
| `SelfDistillDG` / `SCOPELite` | Oracle-label dense-correction toys that study the bridge from sparse rewards to token supervision. They are not learned revisers or process reward models. |

Unsupported regimes are rejected at config validation where the local contract
would silently change the method meaning.
