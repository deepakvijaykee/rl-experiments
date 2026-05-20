# Implementation scope

Every method in this sandbox is a scoped reimplementation of the published method, kept faithful where the update rule, credit-assignment, and normalization choices are concerned, and trimmed where the distributed-system scaffolding around the method does not apply to a single-GPU toy setup. Where the scoping would silently change what the method means, for instance running advantage normalization in a regime where the original paper assumes a critic, the trainer rejects the config rather than running a variant that would be misleading to compare.

| Method family | Scope here |
| --- | --- |
| `GRPO` | Group-normalized rewards, PPO-style clipped token surrogate, optional reverse-KL penalty. Out of scope: a critic, a distributed rollout service, long-response infrastructure. |
| `DrGRPO` | Group-centered rewards without reward-std normalization. Fixed-length tasks remove most observable response-length bias. |
| `DAPOLite` | Group filtering, decoupled clipping, token-level aggregation. Overlong-response reward shaping is omitted because toy responses have fixed length. |
| `TPO` | Sampled-candidate target policy optimization over grouped rollouts, with old-policy anchors and z-scored rewards. |
| `TPOFullAction` | Single-sample full-action classification TPO, scoped to clean on-policy MNIST with one optimizer epoch. |
| `TPOToken` / `GRPOToken` | Per-prefix candidate-simplex objectives for dense token-reward reversal. These consume a dedicated token-candidate batch rather than ordinary rollout batches. |
| `TEMPO` | Toy nonparametric prefix-tree credit within grouped rollouts. Valid only while groups keep a mixed reward signal. |
| `ReplayDG` / `FreshDG` | Replay composition with exponential age weighting. Useful as freshness diagnostics; the buffer is in-process and small, so the implementation is not a production replay system. |
| `UncertaintyDG` / `FilteredDG` / `RewardVarianceDG` | Reward-disagreement heuristics for controlled proxy-noise stress tests. The uncertainty signal is hand-rolled from batch statistics, with no learned verifier-confidence model behind it. |
| `DGEntropyGuard` | Sampled-action probability guard for entropy-collapse diagnostics. Stops short of full covariance control. |
| `SelfDistillDG` / `SCOPELite` | Oracle-label dense-correction toys for studying the bridge from sparse trajectory reward to token supervision. Reviser labels come from the oracle, not from a learned reviser or a process reward model. |
