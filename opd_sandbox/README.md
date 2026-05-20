# opd_sandbox

A small appendix sandbox for on-policy distillation mechanics.

The OPD design space is full of choices that look interchangeable on paper and behave very differently in practice. Which gradient estimator the loss is built on. Which support the truncation runs over. When in training the switch from full vocabulary to truncated support is taken. How sharp the teacher should be. Each of those carries folklore, and the mechanism behind the folklore is hard to test at LLM scale because the runs are slow and the failure modes get masked by confounds from optimization, teacher quality, and pipeline scaffolding.

This package is the cheap, controlled version. Tiny student, full-support smoothed oracle teacher, short horizons. The teacher is an oracle so the design-choice questions stay clean of teacher-quality confounds. The student is small enough that the gradient signal stays readable when something looks wrong. Both choices are about preserving observability, not chasing benchmark performance. What each experiment is after is the failure signature: which diagnostic moves under each design choice, and what its movement says about the mechanism the recipe is sitting on.

## Methods

| Method | Scope |
| --- | --- |
| `OPDReverseKL` | Full-vocabulary `KL(pi_student || pi_teacher)` on student-sampled prefixes. |
| `OPDTopKReverseKL` | Unnormalized reverse-KL contribution restricted to student, teacher, or intersection top-k support. |
| `OPDPG` | Sampled-token reverse-KL reward `log pi_teacher(a|s) - log pi_old(a|s)` with clipped importance sampling. |

## Tasks

- `reversal`: generate the reverse of a random token string.
- `soft_reversal`: reversal with a graded teacher that prefers the correct token, nearby content tokens, and then special tokens.
- `format_answer`: emit an answer tag, modular checksum, then copy suffix tokens. Adds a format-versus-answer axis for OPD diagnostics.

## Reading order

The five experiments build on each other and are intended to be read in order, because each one isolates a precondition for the next.

`variance_microscope.py` comes first because estimator choice is the precondition for any later comparison. The experiment isolates how much horizon-dependent noise each reverse-KL estimator injects into the gradient before optimization is allowed to help, and it does so on a fixed random transformer so the variance number reflects only the estimator. If the estimator is already drowning in noise at short horizons, no design choice downstream can recover.

`topk_stability.py` is next because it is the hostile cold-start case for top-k OPD. With a random student, no SFT alignment, and no same-family teacher, support truncation can remove the only useful signal in the gradient. The run shows the failure mode in its starkest form: the distribution stays near-uniform throughout training, never picking up the teacher's preferred-token asymmetry. There is no sharp wrong mode to diagnose, only a quiet absence of learning.

`topk_cold_start.py` then asks the natural follow-up. If support truncation fails from a random cold start because support overlap is too weak, does a full-vocabulary warmup fix the problem? A 100-step warmup probe is enough to surface a sharper observation: partial warmup can be erased by switching too early, because the truncation kicks in before the teacher's corrective mass has been pulled into the retained support.

`topk_warmup_sweep.py` follows by sweeping the warmup length and pairing it with two truncation widths (`k=4` and `k=8`) that ask structurally different questions. `k=8` is a mild regularizer over a nine-action vocabulary. `k=4` cuts the action space roughly in half. Treating both as a single "top-k" lever averages over a mechanism that does not combine, and the threshold for when truncation becomes stable depends on which one is being used.

`soft_teacher_topk.py` closes the appendix by complicating the story. A smoothed one-hot teacher is artificially crisp; a graded teacher is more realistic. Once the teacher is allowed to be soft, the support-choice question takes on new structure: teacher-support and intersection-support can beat student-support at cold start when the teacher is sharp, and student-support can win after enough warmup. There is no global ordering among the three.

Several losses in this sandbox use unnormalized top-k mass on purpose. The choice is diagnostic. Renormalizing over the retained support would make omitted teacher mass invisible to the loss, and the failure modes the experiments are trying to expose live precisely in that omitted mass.

## What I take from these runs

The experiments are small, but they isolate mechanisms that are easy to lose track of inside a large-model training run. Five inferences I would lead with.

The first is about estimator choice. Cumulative-return REINFORCE puts a sum-over-future-rewards on each token's score function, so both the magnitude of each summand and the number of summed terms grow with horizon. Gradient variance accordingly grows faster than the count of summands alone would predict. The variance microscope shows the gap concretely: at horizon 64, the sequence-level sampled estimator is roughly 10^7 times noisier than the exact full-vocabulary objective on the same prefixes. Per-token OPD is not just a compute-saving change. It changes the stability regime the optimizer sees before any learning has started.

The second is that top-k truncation is a support-overlap assumption that the gradient enforces directly. Reverse KL is an expectation under the student, so every per-action term in the gradient is weighted by `π_student(a)`. Any teacher mass that lies outside the student's top-k contributes exactly zero to the update. From a random cold start, the student's top-k is largely arbitrary, the teacher's preferred token frequently lands outside it, and the optimizer has no mechanism to pull probability toward what it cannot see. The failure mode is unusual: nothing sharp, just a quiet absence of learning that looks like very slow progress.

The third is that warmup only helps when it changes behavior. Loss reductions during the full-vocabulary phase do not by themselves create the support overlap that top-k needs, and switching too early can erase whatever behavioral asymmetry the warmup had produced. The diagnostics that actually predict stable switching are behavioral, meaning top-1 agreement and sampled reward, read together with entropy and teacher mass on the selected support. No single overlap number suffices on its own.

The fourth concerns the width of top-k. In a nine-action toy task, `k=8` excludes one action while `k=4` covers less than half the action space. Treating both as the same "top-k" intervention hides the mechanism and produces overbroad conclusions about when truncation works.

The fifth is that teacher entropy is a first-order variable. A broad teacher can be argmax-correct and still be a weak exact-token teacher at fixed compute, because the directional component of the reverse-KL gradient is proportional to how much `log π_teacher(a)` varies across the support. Soft support makes top-k less brittle in some regimes and dilutes the corrective signal in others, so the right teacher temperature depends on which reward metric is being optimized.

Pulling these together: top-k OPD becomes meaningful when three preconditions are in place, namely that the student already occupies a support containing the teacher's useful tokens, that the teacher is sharp enough on that support to produce a meaningful gradient, and that the truncated objective still carries the correction the full teacher would have provided. The runs in this appendix probe where each precondition can fail, and they map out the boundary of when top-k is the right tool to reach for.

## Quick runs

```bash
python -m opd_sandbox.train --task reversal --method OPDReverseKL \
  --batch_size 64 --num_steps 300 --eval_every 20

python -m opd_sandbox.train --task reversal --method OPDPG \
  --batch_size 64 --num_steps 300 --eval_every 20
```

These two commands cover the basic exact-versus-sampled comparison on `reversal`. The package also supports `OPDTopKReverseKL` with student/teacher/intersection top-k support, the `soft_reversal` and `format_answer` tasks, and teacher-entropy sweeps. The experiment scripts under `experiments/` set those up, and the analysis docs linked below interpret each run.

## First comparison figure

```bash
python -m opd_sandbox.compare --task reversal \
  --batch_size 64 --num_steps 300 --eval_every 20 --num_seeds 3
```

This writes per-method CSVs, a combined CSV, and `opd_results/reversal_opd_compare.png`.

## Variance microscope

The first appendix experiment measures estimator variance without doing any training. It samples student rollouts from a fixed tiny transformer, computes three OPD-style gradient estimators on the same batches, and reports the variance of a fixed random gradient projection as horizon grows.

- `sequence_pg`: cumulative sampled-return score estimator.
- `token_pg`: one-step sampled score estimator with gamma=0 credit.
- `full_vocab_rkl`: exact per-token `KL(pi_student || pi_teacher)`.

```bash
python -m opd_sandbox.experiments.variance_microscope \
  --horizons 4,8,16,32 --num_batches 40 --num_seeds 3
```

This writes `opd_results/variance_microscope.csv` and `opd_results/variance_microscope.png`. The interpreted run is in [`analysis/variance_microscope.md`](analysis/variance_microscope.md).

## Top-k stability

The second appendix experiment compares full-vocabulary reverse KL, student-top-k truncated reverse KL, and sampled-token OPD on the same toy task.

```bash
python -m opd_sandbox.experiments.topk_stability \
  --top_ks 1,2,4 --num_steps 300 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

This writes `topk_stability.csv`, per-variant CSVs, and `topk_stability.png` in the output directory. The interpreted cold-start support-overlap stress test is in [`analysis/topk_stability.md`](analysis/topk_stability.md).

## Top-k cold start

The follow-up that tests whether a full-vocabulary warmup makes top-k truncation stable:

```bash
python -m opd_sandbox.experiments.topk_cold_start \
  --top_ks 1,2,4 --num_steps 300 --warmup_steps 100 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted run is in [`analysis/topk_cold_start.md`](analysis/topk_cold_start.md).

## Top-k warmup sweep

The sweep over warmup length before switching to top-k:

```bash
python -m opd_sandbox.experiments.topk_warmup_sweep \
  --top_ks 4,8 --warmup_steps 0,50,100,150,200,250 \
  --num_steps 300 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted support-threshold result is in [`analysis/topk_warmup_sweep.md`](analysis/topk_warmup_sweep.md).

## Soft-teacher top-k support

The follow-up that swaps the smoothed one-hot oracle for a graded teacher and compares student, teacher, and intersection top-k support:

```bash
python -m opd_sandbox.experiments.soft_teacher_topk \
  --top_ks 4 \
  --supports student,teacher,intersection \
  --warmup_steps 0,100,200,250 \
  --num_steps 300 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted support-choice result is in [`analysis/soft_teacher_topk.md`](analysis/soft_teacher_topk.md).
