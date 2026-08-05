# Policy–Environment Learnability

Sparse terminal reward can train a policy only when the current policy, the
environment, and the rollout allocation jointly produce contrasting outcomes.
This sandbox studies that boundary before comparing optimizers.

The distinction matters across this repository. `rl_sandbox` asks how an
update rule weights a collected batch. This sandbox asks whether the batch
contains a task direction and whether the policy tokens receiving credit could
have changed the outcome. The same distinction organizes the relationship
between pretraining, SFT, distillation, and online RL: each stage changes which
successful trajectories the next stage can see and learn from.

## The reward-contrast boundary

Consider a task with a unique successful path and `D` uncertain decisions. If
the policy selects the correct action with probability `p_t` at decision `t`,
the probability of a successful trajectory is

```text
q = product_t p_t.
```

For the homogeneous analytical environment, `p_t = p` and `q = p^D`. A
group-centered estimator with `K` trajectories receives a nonzero task
direction only when the group contains both successes and failures. The
probability of that event is

```text
M(q, K) = 1 - q^K - (1 - q)^K.
```

This produces three learning regimes:

- **Cold:** `qK << 1`. Groups are almost always all-failure, so an on-policy
  outcome reward supplies no direction.
- **Contrastive:** success and failure coexist within groups, so relative
  advantages can identify behavior associated with reward.
- **Saturated:** `(1 - q)K << 1`. Groups are almost always all-success and the
  centered reward again supplies no direction.

The useful object is therefore a distribution over task-level `q`, not one
aggregate accuracy. Two policies with the same pass@1 can place very different
fractions of tasks inside the contrastive regime.

### Group size under a fixed rollout budget

Increasing `K` raises the probability that any one prompt group contains
contrast. It also concentrates a fixed rollout budget on fewer prompts. With
`B` trajectories arranged into groups of size `K`,

```text
expected successful trajectories = B q
expected mixed prompt groups = (B / K) M(q, K).
```

Group size reorganizes the available successes; it does not create them. A
larger group exposes more trajectories to a within-prompt baseline while
reducing independent prompt coverage. This is why group size, curriculum, and
SFT cannot be treated as interchangeable remedies for sparse reward.

## How training stages move the boundary

The usual stage names hide the mechanism. Their roles become clearer when
expressed through support, decision depth, and reward contrast.

| Intervention | Primary effect on the next stage |
|---|---|
| More capable pretraining | Raises conditional competence and places useful actions in support |
| Reasoning-trace SFT | Raises trajectory support across intermediate decisions and teaches the interaction format |
| Answer-only SFT | Can improve the final-choice mode without creating comparable trajectory diversity |
| Curriculum or task decomposition | Reduces effective decision depth and moves cold tasks into the contrastive regime |
| Larger rollout groups | Changes within-prompt contrast and prompt coverage at fixed rollout cost |
| Off-policy distillation | Injects useful actions that the student would rarely or never sample |
| On-policy distillation | Transfers teacher structure on student-visited states with lower distribution mismatch |
| Online RL | Exploits reward contrast already present under the current policy and environment |

This interpretation reconciles SFT-dependent RL with RL-zero. SFT is one way
to cross the cold boundary. A sufficiently capable pretrained policy, large
rollout budget, curriculum, task decomposition, or exploration mechanism can
cross it without a separate SFT stage. Off-policy teaching is uniquely useful
when correct behavior is absent from student rollouts; on-policy objectives
cannot assign positive credit to a mode they never observe.

The upper boundary matters too. When a task becomes all-success within groups,
more updates on the same binary reward stop carrying task information. Harder
curricula, more granular verification, or a different task distribution then
create more signal than another update on saturated prompts.

## Analytical environment

[`env.py`](env.py) constructs a tabular unique-path task with a fixed action
count and an exactly initialized correct-action probability. One wrong action
terminates the trajectory; completing the path yields reward one.

[`train.py`](train.py) applies a trajectory policy gradient with a
within-prompt baseline — mean-centered by default, or GRPO's std-normalized
advantages via `--advantage_normalization std`. It records exact policy
probabilities beside the sampled group statistics, keeping sampling error
separate from the mechanism. Both estimators preserve the presence or
absence of the GRPO task direction without importing clipping, KL, or
distributed rollout machinery; Result 3 measures how they differ.

```bash
python -m learnability_sandbox.train \
  --horizon 3 \
  --initial_correct_probability 0.5 \
  --group_size 8 \
  --groups_per_step 64 \
  --num_steps 200 \
  --num_seeds 3 \
  --output results/learnability_h3.csv
```

The central measurements are terminal success, pass@`K`, predicted and sampled
mixed-group rates, state reach probability by depth, and expected visited
steps.

## Chess topology from the paper

[Understanding Reasoning from Pretraining to
Post-Training](https://arxiv.org/abs/2607.16097v1) studies how pretraining scale
changes subsequent SFT and RL in a controlled chess pipeline. The paper reports
that pretraining loss predicts post-RL performance and that the local RL slope
grows with pretraining tokens. It also shows heterogeneous policy evolution:
RL amplifies already-preferred correct moves on easier states, surfaces some
buried correct moves on harder states, and sometimes reinforces wrong modes.

This sandbox uses that setup to ask a mechanistic question underneath the
scaling relationship:

> Does pretraining and SFT make RL effective by moving more tasks into the
> reward-contrast region, and does that mediator explain when RL sharpens,
> discovers, or amplifies the wrong mode?

[`chess_env.py`](chess_env.py) implements the strict puzzle interaction
described in the paper and the teacher-forced reply behavior in the released
RL worker. It delegates UCI parsing, legality, and board transitions to
`python-chess`.

[`paper_chess.py`](paper_chess.py) runs the controlled reward-availability
experiment on the public
[`chess-rl-data-balanced@022e7bbe`](https://huggingface.co/datasets/chess-pre-to-post/chess-rl-data-balanced/tree/022e7bbe9ff36b58299ec44f8da08f8324ef5330)
dataset. This balanced set is an environment-topology panel; the paper's
reported RL runs used a different easy-skewed training distribution. The
released source is frozen at
[`pavelslab-nyu/pre2post-chess@256e8b64`](https://github.com/pavelslab-nyu/pre2post-chess/tree/256e8b64d1c4b331e6d327c281169ce4959235c4).

The released puzzle contract is:

- `FEN` is the position before the opponent's trigger move;
- `Moves[0]` is the trigger;
- `Moves[1::2]` are solver actions;
- `Moves[2::2]` are opponent replies;
- `reward_model.ground_truth` is the solver-action sequence.

Every puzzle is validated when loaded. Reset applies the trigger move. Under
`strict_termination`, a malformed, illegal, or incorrect solver action ends
the trajectory with reward zero. A correct action either applies the recorded
opponent reply or completes the puzzle with reward one.

The controlled policy chooses the target with probability `p` whenever an
alternative legal move exists; forced states are always correct. The relevant
exponent is decision depth `D`, rather than raw solution horizon `H`.

```bash
python -m learnability_sandbox.paper_chess \
  --input_path /path/to/train_v4_dataset_balanced_multi_turn.parquet \
  --correct_probability 0.6 \
  --group_size 8 \
  --num_seeds 3 \
  --protocol both \
  --output_path results/paper_chess_protocol_ablation_p06_k8.csv
```

## Result 1: topology preserves the analytical boundary

All 53,225 public puzzles satisfy the move-line and terminal-reward contract.
They contain 133,737 solver states across horizons one through six; 469 states
have only one legal action.

The table compares exact predictions with empirical means across three seeds
at `p=0.6`, `K=8`.

| Horizon | Puzzles | Mean `D` | Exact reward | Sampled reward | Exact mixed | Sampled mixed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,495 | 1.000 | 0.6000 | 0.5996 | 0.9825 | 0.9821 |
| 2 | 23,226 | 2.000 | 0.3600 | 0.3609 | 0.9716 | 0.9724 |
| 3 | 18,314 | 2.986 | 0.2181 | 0.2180 | 0.8589 | 0.8587 |
| 4 | 4,484 | 3.972 | 0.1321 | 0.1324 | 0.6757 | 0.6762 |
| 5 | 1,324 | 4.952 | 0.0805 | 0.0802 | 0.4860 | 0.4829 |
| 6 | 382 | 5.953 | 0.0482 | 0.0501 | 0.3253 | 0.3360 |

The analytical law survives the real puzzle topology. Forced moves alter the
exponent only rarely. At horizon six, a policy that is 60% correct at each
uncertain decision succeeds on roughly 5% of trajectories, and about two
thirds of eight-sample groups carry no task direction.

This is the first bridge to the paper's pretraining-to-RL result. A stronger
initial policy compounds through the trajectory and changes the supply of
contrastive groups nonlinearly. Aggregate pass@1 alone hides that mechanism;
the task-level distribution of success probability determines how many prompts
can contribute to an online update.

## Result 2: reward availability and causal credit are distinct

The paper describes immediate termination after an incorrect move. The
released
[`fsdp_workers.py`](https://github.com/pavelslab-nyu/pre2post-chess/blob/256e8b64d1c4b331e6d327c281169ce4959235c4/rl/verl/workers/fsdp_workers.py)
instead consumes the next recorded opponent reply whenever the model calls the
environment. The released
[`reward_function_multiturn.py`](https://github.com/pavelslab-nyu/pre2post-chess/blob/256e8b64d1c4b331e6d327c281169ce4959235c4/rl/verl/reward_function_multiturn.py)
checks the complete solver sequence afterward. The corresponding adapter here
is named `teacher_forced_reply_replay`.

For a fixed action plan with per-step correctness probabilities `p_t`, both
protocols award one exactly when every solver action is correct:

```text
q = product_t p_t.
```

They therefore have the same terminal reward and mixed-group probability.
Their trajectory distributions differ. Strict termination reaches step `t`
only when the earlier prefix is correct:

```text
L_strict = sum_t product_{j<t} p_j.
```

Reply replay always collects `H` solver actions. The difference consists of
actions sampled after the first error, when terminal success is already
impossible:

```text
L_replay - L_strict = H - L_strict.
```

`paper_chess.py` samples one complete `K` by `H` action plan for each puzzle and
supplies that same plan to both environments. Policy randomness is paired, so
the comparison isolates the transition protocol.

![Stacked bars showing strict trajectory length and additional post-error actions under teacher-forced reply replay across puzzle horizons one through six.](figures/teacher_forced_reply_ablation.png)

*Reply replay preserves the full horizon by adding actions after strict
termination would have ended the trajectory. Percentages denote the fraction
of replay actions generated after the first error.*

| Horizon | Strict actions | Replay actions | Replay post-error actions |
|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 0.0000 |
| 2 | 1.6000 | 2.0000 | 0.4000 |
| 3 | 1.9628 | 3.0000 | 1.0372 |
| 4 | 2.1813 | 4.0000 | 1.8187 |
| 5 | 2.3158 | 5.0000 | 2.6842 |
| 6 | 2.3925 | 6.0000 | 3.6075 |
| All puzzles | 1.7354 | 2.5127 | 0.7773 |

Terminal reward and mixed-group rate match exactly in every paired
seed-by-horizon aggregate. Across the dataset, replay increases sampled solver
actions by 44.8%; 30.9% of its solver actions occur after the first error. At
horizon six, that fraction reaches 60.1%.

The released worker masks injected environment replies while retaining later
model tokens in the policy mask. Replay therefore assigns a failed
trajectory's sequence-level advantage to actions generated after failure was
already inevitable, under a context that combines a model error with a reply
from the incompatible reference line. It changes the support and causal
meaning of the update while preserving reward availability.

The controlled result establishes the amount and location of that extra
sampling. Its effect on learning is the next causal question.

## Result 3: the drift law behind the reservoir

The reservoir picture needs one local law: under group-centered credit the
expected success-probability drift of a task obeys `q̇ ∝ q²` at small `q`.
Both advantage estimators carry it — the mean baseline because the unbiased
gradient itself is proportional to `q`, std normalization because rescaled
mixed groups arrive at rate `Kq` while the gradient supplies the other
factor — and they differ only by a rate constant of order `√K`.
[`analytical_drift.py`](analytical_drift.py) measures this by direct
simulation with per-prompt SGD, so the estimator's own expectation is what
appears, not an adaptive-optimizer artifact.

| Measurement | Prediction | Measured |
|---|---|---|
| `log q̇` vs `log q₀` slope, mean baseline | 2 | 1.94 / 1.96 / 1.95 (K=4/8/16) |
| same, std normalization | ≈2 | 1.87 / 1.87 / 1.79 |
| std/mean drift ratio at fixed task | grows as `√K` | 2.4 / 2.7 / 3.2 (K=4/8/16) |
| conversion-time ratio per halving of `q₀`, cold end | 2 | 1.91–1.92 (mean), 1.85–1.89 (std) |

The equal-weight mixture over depths 1–8 sweeps the log-linear window
predicted by claim 1: conversion times spaced geometrically in `q₀` yield
reward approximately linear in `log` steps until the deepest tasks deplete.
Warm starts convert faster than the asymptote (pooled power-law exponent
0.8 rather than 1), which is the `D(1-p)²` prefactor plus band proximity,
not a violation of the cold-end law.

![Three panels: early drift against initial success probability with slope-2
reference, conversion time against inverse initial success probability for
both advantage estimators, and the depth-mixture success curve against log
training step showing the log-linear window.](figures/analytical_drift.png)

The practical consequence: advantage normalization moves the rate constant,
not the band structure or the curve shape. The paper's log-linear form is
robust to the estimator variant; what discriminates estimators is the
`K`-dependence of the slope, which the group-size sweep can measure.

```bash
python -m learnability_sandbox.analytical_drift
```

## Result 4: thinking breaks the teacher-forced instrument on the RL axis

Under the multi-turn protocol, opponent replies are injected, so
`exp(-line_nll)` — the teacher-forced probability of the solver line, now
emitted per puzzle by `coordinate_eval` — would equal the task success
probability `q` exactly *if the policy emitted moves directly*. Pretraining
checkpoints do. The SFT/RL rollout grammar does not: the model generates a
latent think block (about 700 tokens at horizon 1) before each move, so
sampled success marginalizes over think paths, and `exp(-line_nll)` is only
the *reflex* success probability, the direct-move mode's share.

Joining the step-1000 RL checkpoint's reflex q̂ against its own n=16
rollouts on the reconstructed B1–B4 panel
([`qhat_validation.py`](qhat_validation.py), 1,160 puzzles):

| | reflex q̂ | rollout success | Spearman | above 99.9% binomial envelope |
|---|---:|---:|---:|---:|
| all | 0.020 | 0.208 | 0.57 | 35.9% |
| B1 | 0.040 | 0.425 | 0.56 | 64.3% |
| B4 | 0.008 | 0.035 | 0.28 | 10.1% |

No puzzle falls significantly below the envelope. A ten-fold thinking lift
with rank correlation decaying on harder bins means the think block carries
real computation even in a 20-million-parameter chess model — puzzles with
reflex q̂ near `1e-7` reach 60% sampled success.

![Scatter of empirical rollout success against reflex success probability
with the binomial envelope, and the decile calibration curve sitting an
order of magnitude above the diagonal.](figures/qhat_validation_rl20m_a0400_step1000.png)

This is the strongest possible form of the claim-2 caveat, and it lands in
the sandbox's favor: chess-SFT is structurally isomorphic to math — latent
reasoning plus a verifiable terminal answer — while chess-pretrain remains
the exact regime. The math-proxy question can therefore be rehearsed
entirely inside chess, against ground truth.

## Result 5: teacher-forced coordinates invert across the pipeline's format boundary

A correction discovered by the instruments themselves: the released
`20m_C_6p5e18_alpha0.400/final` checkpoint ships the 81-token *pretraining*
vocabulary — it is the pretrained base, not an SFT model, and the paper
released no standalone SFT checkpoints at all (the nearest post-SFT
artifacts are RL step 50 in the verl runs and step 20 in the miles runs).
The measured series is therefore base → RL steps, with the SFT stage
folded into the first hop
([`transfer_shells.py`](transfer_shells.py), full B1–B5 panel, shells
fixed at the base's *exact* q̂ — the pretrain format has no think block,
so the reflex instrument is exact there by construction):

| coordinate | pretrain base | RL 50 | RL 1000 | RL 5000 |
|---|---:|---:|---:|---:|
| total CE | 0.476 | 1.045 | 1.028 | 1.115 |
| decision CE | 1.76 | 3.22 | 3.25 | 3.94 |
| mean reflex q̂ | 0.099 | 0.010 | 0.016 | 0.012 |
| reflex contrast mass (K=8) | 0.306 | 0.066 | 0.091 | 0.070 |

Two boundaries, two behaviors. Crossing the format boundary (base to
step 50, which spans SFT plus fifty RL steps) collapses reflex q̂ by an
order of magnitude in every shell — the think gate now sits in front of
every move, so the teacher-forced line measures format preference, not
competence change. Within RL, the coordinates move non-monotonically:
reflex q̂ recovers slightly to step 1000 while published pass@1 rises,
then decision CE degrades sharply from 3.25 to 3.94 by step 5000 — the
late-RL reference-line drain. Median reflex q̂ ends 1.5–1.9 decades below
the base in every shell, including the 528-puzzle band shell whose exact
base contrast mass was 0.72.

![Shell medians of reflex q̂ across the base-to-RL series, all falling
across the format boundary, beside the per-checkpoint reservoir histograms
shifting left with a growing deep tail.](figures/transfer_shells_rl20m_a0400.png)

Two readings must be kept apart. This measures the *reflex* distribution's
evolution, not task success: the diagonal-approximation test (does RL move
tasks it never sampled successes on?) still requires sampled q̂ at several
RL steps. But as an instrument verdict it is final — checkpoint
coordinates are exact on the pretrain format, format-dominated the moment
the think grammar appears, and non-monotone under RL, so any coordinate
regression mixing formats through teacher-forced loss compares quantities
that move in opposite directions. Off the pretrain axis, coordinates must
be computed on sampled trajectories. The immediate refinement is a
gate/aim decomposition of each move span (first-token NLL separates the
think-or-move gate from move identity), which the same forward pass can
emit; the missing SFT step-0 requires training our own SFT from the
released `sft_v1_200m_90k` corpus — infrastructure the decoupling fork
needs regardless.

## Result 6: pass@k stagnation above the training group size, in the paper's own curves

pass@k is a moment transform of the task-level success distribution,
`pass@k = 1 - E[(1-q)^k]`, so the released run grid's published pass@k
columns carry a coarse image of the q-histogram evolving — no model
evaluation required.
[`released_curve_analysis.py`](released_curve_analysis.py) reads all 14
released RL runs (20m–680m, alpha 0.05–1.0, every run trained at K=8) from
`rl_curve_points.csv` and measures three things.

**Gain profiles.** From first to last eval step, pass@1 gains run +0.02 to
+0.09 across runs while the median pass@16/pass@1 gain ratio is **-0.05**:
pass@16 simply does not improve. Only one run clears two (conservative,
unpaired) standard errors at k=16 — and it is the exception the theory
predicts: the weakest-pretrained 200m run (alpha 0.05, gain +0.026), whose
success distribution sits low enough that the K=8 band overlaps the region
pass@16 is still sensitive to. Band conversion shows up at k=16 exactly
when the band has not yet outrun it.

**Tail-difference depletion.** `pass@16 - pass@8` isolates the mass in the
half-decade below the band. It declines in **14 of 14 runs**. Diagonal
conversion drains this band from its upper edge; refill could only come
from deep-cold tasks rising by off-diagonal generalization. Net depletion
everywhere means transfer nowhere dominates drain — at any of the four
model scales. This is the observational form of the
diagonal-approximation test, and it bounds the correction term the
reservoir theory omits.

**Shell deconvolution.** Constrained NNLS on five fixed atoms inverts the
four moments into shell masses per eval step. The 20m alpha 0.4 run shows
conveyor structure: saturated mass rises, the cold shell drains, deep-cold
mass stays frozen, and the band holds a quasi-steady level — inflow from
the cold edge balancing conversion — which is the steady-state reservoir
sweep of claim 1 seen in published aggregates.

![Three panels: pass@k gain against k for every released run collapsing to
zero at k=16, the pass@16 minus pass@8 tail mass declining across training
in all runs, and deconvolved shell masses for the 20m alpha 0.4 run showing
saturated mass rising while cold mass drains.](figures/released_curve_claim3.png)

Caveats kept in view: the released curves stop at step 1000, so reservoir
depletion past the published window (claim 1's curvature) still needs the
local step-5000 checkpoints; and four moments cannot localize the histogram
finely — the shell reading is coarse by construction, and the gain profile
is the primary evidence.

```bash
python -m learnability_sandbox.released_curve_analysis
```

## Result 7: the released SFT sweep is a mid-training dose-response experiment

The released `all_sft_models.csv` grid contains the closest existing
off-manifold axis to the headline question: eight pretrained bases carry
SFT models at two SFT compute fractions each, and nineteen bases carry both
`thinking` and `nonthinking` SFT variants at matched dose — all with
published pass@k.
[`sft_dose_analysis.py`](sft_dose_analysis.py) applies the Result 6
deconvolution to that grid.

**Dose-response.** Contrast mass at K=8 rises with SFT dose in 8 of 8
bases (median +0.18): on the released dose range, adaptation is still
lifting cold mass into the band. The sub-band tail (`pass@16 - pass@8`,
model-free) shows the predicted turnover: it grows with dose in all four
50m bases (small doses, weaker models) and *shrinks* with dose in three of
four 200m bases (larger doses, stronger models), with the loss growing
monotonically in pretraining allocation (-0.008 at alpha 0.2 to -0.021 at
alpha 1.0). Adaptation first fills the near-band reservoir, then begins
draining it — the non-monotone dose-response that a mid-training stop rule
exists to catch, visible in released aggregates.

**Format contribution.** Thinking minus nonthinking at matched base and
dose: the pass@1 gap straddles zero (median +0.007, negative for every
weakest-alpha base), but the sub-band tail gap is positive in **18 of 19
pairs** (median +0.034). At the SFT stage the think format buys little
immediate accuracy; what it buys is tail thickness — mass in exactly the
region RL at K=8 consumes next. The think format is a reservoir-shaping
intervention: it supplies RL fuel, not SFT performance. The gap is largest
for the thin-SFT high-compute 200m pair (+0.07 to +0.09 pass@1, the
mid-training-like corner of the grid) and negative only for the weakest
200m base.

![Three panels: deconvolved contrast mass rising with SFT dose in every
base, the model-free sub-band tail rising for 50m bases and falling for
200m bases, and the thinking-versus-nonthinking scatter with nearly all
pairs above zero on the tail axis while straddling zero on
pass@1.](figures/sft_dose_response.png)

Caveats: two doses per base give direction, not curvature; dose ranges
differ by size, so the turnover is a dose-by-strength interaction read
across bases rather than a within-base reversal; the deconvolution is
coarse, which is why the model-free tail moment carries the finding. The
nonthinking checkpoints also matter beyond this comparison: they emit
moves directly, so the reflex instrument is exact on them, making them the
end-to-end certification target for the teacher-forced coordinate chain
against published pass@k.

```bash
python -m learnability_sandbox.sft_dose_analysis
```

## The headline question: mid-training + RL

The paper's law is fitted along a from-scratch manifold where model size and
tokens parameterize everything. The regime practitioners actually live in —
a large generically-trained model plus a comparatively thin domain
adaptation — breaks that parameterization: domain loss decouples from
general capability, and the paper's coordinates stop being well-defined.
The headline question of this sandbox is therefore: **which checkpoint-local
coordinate predicts RL's payoff regardless of training history, and what
stop rule for domain adaptation follows from it?** Two halves, one per
domain: the *decoupling fork* in chess, where the candidate coordinate
(decision-entropy-weighted loss) is exact; and the *math transfer* in the
paper's OLMo-2 / NuminaMath / GSM8K+MATH setting, the realistic regime,
where the chess result predicts which proxy weighting transfers. The theory
below is the engine room serving this question; its claim 2 is the
headline's theoretical core, and the fork's training runs do not start until
their quantitative predictions are pre-registered from that theory. The
sharpest pre-registrable form is a **crossing prediction**: two mid-trained
checkpoints matched on total domain loss but differing in band mass should
show RL curves that cross — the band-heavy branch wins at small RL compute,
the deeper-reservoir branch at large — which the paper's coordinates forbid,
since `f` and `g` co-move along the from-scratch manifold.

## A first-principles program

The paper reports four regularities it does not derive: a log-linear RL
curve, a slope carried by pretraining tokens, pass@k stagnation, and a
compute-optimal RL fraction that grows with budget. One theoretical object
plausibly generates all four: the task-level success-probability
distribution `q` evolving under a band-limited learning operator — on-policy
group RL moves mass only inside the contrastive band `M(q, K)`. The program
is to write that object down exactly (the analytical environment computes
`q̇` in closed form), derive the regularities as consequences, and test each
derived consequence on the released chess artifacts and on the paper's math
setting. Five claims, each falsifiable:

1. **The log-linear law is a window artifact.** Depth-heterogeneous tasks
   (`q = p^D`) are log-uniformly spaced in `q`, so each decade of RL compute
   converts a near-constant task mass: log-linear reward, with slope equal
   to task density per decade near the band. Predicts sub-log-linear
   curvature when the reservoir depletes — checkable in the released
   checkpoints past step 1000, which the published curves do not reach.
2. **Loss is the wrong coordinate off the from-scratch manifold.** Loss
   mixes calibration with decision competence; mid-training decouples them.
   The statistic that should transfer across training histories is
   decision-entropy-weighted loss (exact in chess: weight by legal-move
   branching), with contrast mass as the checkpoint-local mediator. This is
   the mid-training + RL question in measurable form: if the coordinate is
   checkpoint-local, no new scaling law is needed for mid-trained models.
3. **Pass@k cannot durably improve for k above the training group size.**
   Group-centered credit only reaches moves sampled within the group
   (`π ≳ 1/K`), while sharpening drains the tail below it. The paper's
   K=8 runs improving pass@8 but not pass@16 is one point; the sweep over
   K is the test.
4. **The allocation trend is a crossover condition.** Pretraining buys
   `dq/dT` everywhere; RL buys `dq/dC` inside the band. Marginal equality
   derives the observed 5–28% RL fraction and predicts its asymptote from
   the difficulty-tail thickness. For mid-training the same condition yields
   a stop rule: end domain adaptation when decision-weighted loss plateaus,
   not total loss.
5. **Wrong-mode amplification is partly a credit artifact.** Teacher-forced
   replay assigns a failed trajectory's advantage to post-error tokens in
   reference-line states — largest exactly on hard puzzles (60% of actions
   at horizon 6). Predicts strict termination or post-error masking reduces
   wrong-mode amplification on B3–B5 at matched compute.

Measured status. Claim 1's local law is now measured (Result 3): drift
slope 2 in `q₀` for both advantage estimators, `√K`-ordered rate constants,
cold-end conversion doubling per halving of `q₀`, and the log-linear
mixture window — the derivation can therefore drop the `p^D` toy form and
state the reservoir on the measured q̂ histogram: the reward curve is
log-linear exactly where the histogram is flat per decade. Claim 2 acquired
an estimator clause (Results 4 and 5): on the SFT/RL vocabulary the
teacher-forced reference line measures the *reflex* mode, which RL drains
while sampled success rises, so off-manifold coordinates must be stated on
sampled trajectories (or a validated cheap surrogate); which surrogate
tracks sampled q̂ is now an internal chess question that rehearses the math
proxy question exactly. Claim 3 is observationally confirmed at K=8 on the
full released run grid (Result 6): pass@16 gains vanish within noise in 13
of 14 runs, the exception being the weakest-pretrained model, where the
band still overlaps pass@16's sensitive region — and the tail-difference
depletion in every run bounds off-diagonal transfer below diagonal drain
at all four model scales. The group-size sweep remains the causal test.
Claim 4's mid-training reading has its first off-manifold evidence
(Result 7): on the released SFT-dose axis, adaptation lifts contrast mass
everywhere while the sub-band tail turns from growing to draining as dose
and base strength rise — the non-monotonicity the stop rule exists to
catch — and the think format's measured contribution at the SFT stage is
tail mass, not accuracy.

Both of the paper's domains serve the program. Chess is the exact
instrument: released checkpoints, corpora, curves, and a branching factor
that makes decision-weighted loss computable without approximation. Math
(the paper's OLMo-2 / NuminaMath / GSM8K+MATH setting) is the transfer
test, and the only domain where the mid-training regime is realistic:
decision weighting needs a proxy there (answer-span restriction or
verifier-step entropy), and claim 2 predicts which proxy works — the one
that tracks chess's exact version.

## Superseded hypotheses

An earlier framing posed four hypotheses (contrast mediation, support
quality, replay credit, phase-dependent interventions) as deltas against
the paper's own frame. They survive inside the claims above: mediation and
support quality inside claim 2, replay credit inside claim 5, phase
dependence inside claims 1 and 4.

## Experiment sequence

Ordered to serve the headline; derivations precede the measurements they
gate, and cheap preparation runs in parallel where no dependency binds.

0. **Instrument certification** (B1–B4 complete, B5 in flight): n=16
   multi-turn rollouts of the released step-1000 RL checkpoint on the
   reconstructed panel, released and board scorers side by side; the
   published-number comparison closes when B5 lands. Result 4's calibration
   join is part of this certification.
1. **Coordinate measurement** (chess, released artifacts): total, decision,
   and entropy-weighted CE per released checkpoint, regressed on published
   per-run RL slopes. `coordinate_eval` now also emits per-puzzle
   `line_nll`, so every coordinate CSV carries the reflex q̂ histogram and
   contrast mass; the 25-checkpoint re-run with the new column is pending.
   Interim observation (50m axis, n=5): all coordinates correlate with the
   slope near-identically (about -0.91) — on the from-scratch manifold
   calibration and competence co-move, so on-manifold data cannot separate
   the coordinates. This is claim 2's predicted collinearity and makes the
   fork necessary rather than merely motivated. Result 5 adds a caution:
   any regression mixing SFT and RL checkpoints through teacher-forced
   loss mixes quantities that move in opposite directions.
2. **Near-SFT q̂ calibration** (gates the fork's instrument): the paper
   released no standalone SFT checkpoints (Result 5's correction), so the
   nearest post-SFT artifact is RL step 50; its n=16 calibration join
   (running) extends Result 4 to the start of RL, and comparing it against
   the step-1000 join measures how calibration degrades along RL. The
   exact step-0 requires training our own SFT from the released
   `sft_v1_200m_90k` corpus — fork infrastructure in any case. If the
   near-SFT reflex is calibrated, the fork pre-registers on cheap
   teacher-forced histograms; otherwise on sampled q̂ (pass@n panels), with
   the gate/aim decomposition (first-token NLL of each move span separates
   the think-or-move gate from move identity) as the candidate cheap
   surrogate. The exact counterpart runs on the released *nonthinking*
   SFT checkpoints (Result 7), which emit moves directly: predict their
   published pass@k from teacher-forced q̂ histograms alone — an
   end-to-end certification of the coordinate chain with no rollouts.
3. **Calibration/competence derivation** (gates the fork): the drift law
   and the log-linear window are measured (Result 3); what remains is the
   fork's pre-registered numbers — the decision-CE gap at matched total CE
   and the GRPO-slope *ratio* between branches implied through contrast
   mass. Ratios, not absolutes: optimizer and step-size constants cancel
   between branches sharing architecture and RL configuration.
4. **Decoupling fork** (headline, chess): two continued-pretraining
   branches — generic vs domain-skewed corpus — stopped at matched total
   domain loss; compare decision-weighted loss, contrast mass, pass@k, then
   small-GRPO slopes against the pre-registered predictions, with the
   crossing prediction as the headline form. Corpus preparation is plumbing
   and proceeds immediately.
5. **Math transfer** (headline, realistic regime): begins with preparation
   alongside the chess fork, not after it — base checkpoints, evaluation
   panels, and candidate decision-weighting proxies (answer-span
   restriction, verifier-step entropy, sampled pass@n histograms); claim 2
   predicts the proxy tracking chess's sampled q̂ transfers.
6. **Reservoir curvature and transfer decomposition** (support): the
   observational form is done — Result 6's tail-difference depletion in
   all 14 released runs bounds off-diagonal transfer below diagonal drain
   through step 1000. What remains: sampled q̂ on a denser checkpoint
   series (per-decision Monte Carlo for tail resolution) for the
   task-level diagonal test, and evaluation of the local step-5000
   checkpoints for the predicted sub-log-linear departure beyond the
   published window. The reflex-based shell analysis (Result 5) tracks the
   wrong distribution on the RL axis; the sampled version is the real
   test.
7. **Group-size sweep** (support): GRPO at K in {4, 8, 32} at matched
   compute — the causal test behind Result 6's observational K=8 result.
   Claim 3 predicts pass@k gains track K, not difficulty alone, and
   Result 3's estimator constants predict how the slope scales with K
   under each advantage normalization.
8. **Credit-protocol intervention** (support): strict vs replay vs
   post-error masking with the policy-evolution taxonomy; claim 5 predicts
   the wrong-mode share moves with the protocol.

## Token-level released-pipeline substrate

The action-level results above abstract away tokens. The modules below
implement the released pipeline's token-level contracts so that checkpoints,
prompts, rollouts, and rewards are exchangeable with the released artifacts.
Two rewards coexist deliberately: the sandbox's canonical reward replays each
rollout's structurally recorded submitted moves through the strict board
environment (`board_verdict`), while the released text-parsing scorer — whose
defects, including castling always converted as White, are preserved — exists
only to compare against released logs and published numbers. Evaluation
reports both, so the released reward's defect rate is a measured quantity.

Parity evidence: the vendored tokenizer matches the frozen released source on
all 53,225 training prompts, all 80,512 recorded environment replies, and
2,000 released eval sequences (encode and decode, zero mismatches), and both
vocabulary layouts match the `vocab.json` shipped inside released
checkpoints; the scorer port reproduces recorded scores on released eval
logs. These parity enumerations and the golden-fixture state-machine suite
run locally against the pinned artifacts. The withheld
B1-B5 test set (1,480 puzzles, bin counts matching the paper) is
reconstructed from released eval logs — teacher forcing prints the true
reply after every `<call_env>` — cross-validated against the open Lichess
puzzle database. Pinned artifact revisions live in the data directory's
manifest.

## Code map

- [`env.py`](env.py): analytical unique-path environment, exact metrics, and
  the group-centered update with mean or GRPO-std advantage normalization.
- [`train.py`](train.py): multi-seed analytical training entry point.
- [`analytical_drift.py`](analytical_drift.py): drift-law measurement —
  `q̇ ∝ q²` scaling, estimator rate constants, conversion times, and the
  log-linear mixture window (Result 3).
- [`qhat_validation.py`](qhat_validation.py): calibration of teacher-forced
  reflex q̂ against sampled rollout success (Result 4).
- [`transfer_shells.py`](transfer_shells.py): shell decomposition of reflex
  q̂ evolution across a checkpoint series (Result 5).
- [`released_curve_analysis.py`](released_curve_analysis.py): claim-3 gain
  profiles, tail-difference depletion, and shell deconvolution on the
  released run grid's published pass@k curves (Result 6).
- [`sft_dose_analysis.py`](sft_dose_analysis.py): adaptation dose-response
  and thinking-vs-nonthinking format contribution on the released SFT
  sweep (Result 7).
- [`chess_env.py`](chess_env.py): strict and teacher-forced puzzle transition
  contracts, and the protocol name registry.
- [`paper_chess.py`](paper_chess.py): paired controlled-policy calibration on
  the released puzzle topology.
- [`lan_tokenizer.py`](lan_tokenizer.py): vendored released LAN tokenizer,
  both vocabulary layouts, move rendering, and the LAN move grammar.
- [`multi_turn.py`](multi_turn.py): token-level multi-turn rollout state
  machine, reply protocols, the canonical board reward, and the released
  parity scorer.
- [`puzzle_data.py`](puzzle_data.py): the token-level puzzle data contract —
  prompt plus validated line, with environment replies derived from the line
  rather than stored beside it — and the parquet loader.
- [`eval_puzzles.py`](eval_puzzles.py): one-time reconstruction of the
  withheld B1-B5 test puzzles from released eval logs and the Lichess
  database.
- [`evaluate_checkpoint.py`](evaluate_checkpoint.py): multi-turn pass@k
  evaluation of released-format checkpoints under either reply protocol.
- [`coordinate_eval.py`](coordinate_eval.py): teacher-forced loss
  coordinates per puzzle — total, decision, entropy-weighted CE and
  `line_nll`, whose exponential is the reflex q̂.
- [`figures/`](figures/): the retained public figures — the protocol
  ablation and the Result 3–7 instrument figures.

Raw runs and heavyweight checkpoints stay outside the public source surface.
The README carries the equations, protocol, final evidence, interpretation,
and next falsifiable experiments needed to understand the work without private
run context.
