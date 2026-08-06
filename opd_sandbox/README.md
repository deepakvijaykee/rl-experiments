# opd_sandbox

A small appendix sandbox for on-policy distillation mechanics.

The OPD design space is full of choices that look interchangeable on paper and behave very differently in practice. Which gradient estimator the loss is built on, which support the truncation runs over, when in training the switch from full vocabulary to truncated support is taken, and how sharp the teacher should be are all decisions with folklore attached, and the mechanism behind that folklore is hard to test at LLM scale. The runs are slow, and the failure modes get masked by confounds from optimization, teacher quality, and pipeline scaffolding.

This package is the cheap, controlled version: a tiny student, a full-support smoothed oracle teacher, and short horizons. The teacher is an oracle so the design-choice questions stay clear of teacher-quality confounds, and the student is small enough that the gradient signal stays readable when something looks wrong. Both choices buy observability, which is the only thing this scale is good for. Each experiment then goes after one failure signature, asking which diagnostic moves under a given design choice and what that movement says about the mechanism the recipe is sitting on.

## Methods

| Method | Scope |
| --- | --- |
| `OPDReverseKL` | Full-vocabulary $\mathrm{KL}(\pi_\text{student} \Vert \pi_\text{teacher})$ on student-sampled prefixes. |
| `OPDTopKReverseKL` | Unnormalized reverse-KL contribution restricted to student, teacher, or intersection top-k support. |
| `OPDPG` | Sampled-token reverse-KL reward $\log \pi_\text{teacher}(a \mid s) - \log \pi_\text{old}(a \mid s)$ with clipped importance sampling. |

## Tasks

- `reversal`: generate the reverse of a random token string.
- `soft_reversal`: reversal with a graded teacher that prefers the correct token, then nearby content tokens, then special tokens.
- `format_answer`: emit an answer tag, a modular checksum, then copy suffix tokens. Adds a format-versus-answer axis for OPD diagnostics.

## Reading order

The five experiments build on each other and are meant to be read in order, because each one isolates a precondition for the next.

`variance_microscope.py` comes first because estimator choice is the precondition for every later comparison. It isolates how much horizon-dependent noise each reverse-KL estimator injects into the gradient before optimization is allowed to help, and it does so on a fixed random transformer so that the variance number reflects the estimator and nothing else. If the estimator is already drowning in noise at short horizons, no design choice downstream can recover.

`topk_stability.py` is next because it is the hostile cold-start case for top-k OPD. With a random student, no SFT alignment, and no same-family teacher, support truncation can remove the only useful signal in the gradient. The run shows the failure mode in its starkest form: the distribution stays near-uniform throughout training and never picks up the teacher's preferred-token asymmetry. There is no sharp wrong mode to diagnose, only a quiet absence of learning.

`topk_cold_start.py` asks the natural follow-up. If support truncation fails from a random cold start because support overlap is too weak, does a full-vocabulary warmup fix the problem? A 100-step warmup probe turns out to be enough to surface a sharper observation, which is that partial warmup can be erased by switching too early, because the truncation kicks in before the teacher's corrective mass has been pulled into the retained support.

`topk_warmup_sweep.py` follows by sweeping the warmup length and pairing it with two truncation widths, $k=4$ and $k=8$, that ask structurally different questions. Over a nine-action vocabulary, $k=8$ is a mild regularizer while $k=4$ cuts the action space roughly in half. Treating both as a single top-k lever averages over a mechanism that does not combine, since the threshold for when truncation becomes stable depends on which one is in use.

`soft_teacher_topk.py` closes the appendix by complicating the story. A smoothed one-hot teacher is artificially crisp, and a graded teacher is more realistic. Once the teacher is allowed to be soft, the support-choice question acquires new structure: teacher-support and intersection-support can beat student-support at cold start when the teacher is sharp, and student-support can win after enough warmup. There is no global ordering among the three.

Several losses in this sandbox use unnormalized top-k mass, which is what keeps them diagnostic. Renormalizing over the retained support would make omitted teacher mass invisible to the loss, and the failure modes these experiments exist to expose live in exactly that omitted mass.

## What I take from these runs

The experiments are small, and what they isolate is easy to lose track of inside a large-model training run. Five inferences I would lead with, in the order the runs establish them.

Estimator choice sets the stability regime everything else operates in, which is why it comes first. Cumulative-return REINFORCE puts a sum over future rewards on each token's score function, so the magnitude of each summand and the number of summed terms both grow with horizon, and gradient variance grows faster than the count of summands alone would predict. The variance microscope puts a number on it: at horizon 64, the sequence-level sampled estimator is roughly 10^7 times noisier than the exact full-vocabulary objective on the same prefixes. Per-token OPD therefore changes what the optimizer sees before any learning has started, which is a much larger claim than the compute saving it usually gets sold on.

Once the estimator is usable, the next thing that can go wrong is support. Top-k truncation reads as a neutral efficiency trick, and the gradient treats it as a support-overlap assumption. Reverse KL is an expectation under the student, so every per-action term in the gradient carries a weight of $\pi_\text{student}(a)$, and teacher mass lying outside the student's top-k contributes exactly zero to the update. From a random cold start the student's top-k is largely arbitrary, the teacher's preferred token frequently lands outside it, and the optimizer has no mechanism for pulling probability toward something it cannot see. The resulting failure is unusually hard to spot. Nothing sharp goes wrong, and the quiet absence of learning looks like very slow progress.

Warmup is the obvious remedy, and it helps only when it changes behavior. Loss reductions during the full-vocabulary phase do not by themselves create the support overlap top-k needs, and switching too early erases whatever behavioral asymmetry the warmup had produced. What predicts a stable switch is the behavioral diagnostics, meaning top-1 agreement and sampled reward, read together with entropy and teacher mass on the selected support. No single overlap number suffices on its own.

Where that switch threshold falls depends on how aggressive the truncation is, which makes the width of top-k two levers wearing one name. In a nine-action toy task, $k=8$ excludes a single action while $k=4$ covers less than half the action space. Treating both as the same intervention hides the mechanism and yields conclusions far broader than the runs support.

Teacher entropy sits underneath all four of those and behaves as a first-order variable. A broad teacher can be argmax-correct and still be a weak exact-token teacher at fixed compute, because the directional component of the reverse-KL gradient is proportional to how much $\log \pi_\text{teacher}(a)$ varies across the support. Soft support makes top-k less brittle in some regimes and dilutes the corrective signal in others, so the right teacher temperature depends on which reward metric is being optimized.

Pulling these together, top-k OPD becomes meaningful when three preconditions hold at once: the student already occupies a support containing the teacher's useful tokens, the teacher is sharp enough on that support to produce a meaningful gradient, and the truncated objective still carries the correction the full teacher would have provided. The runs in this appendix probe where each precondition can fail, which is what maps out the boundary of when top-k is the right tool to reach for.

## Quick runs

```bash
python -m opd_sandbox.train --task reversal --method OPDReverseKL \
  --batch_size 64 --num_steps 300 --eval_every 20

python -m opd_sandbox.train --task reversal --method OPDPG \
  --batch_size 64 --num_steps 300 --eval_every 20
```

These two commands cover the basic exact-versus-sampled comparison on `reversal`. The package also supports `OPDTopKReverseKL` with student, teacher, or intersection top-k support, the `soft_reversal` and `format_answer` tasks, and teacher-entropy sweeps. The experiment scripts under `experiments/` set those up, and the analysis docs linked below interpret each run.

## First comparison figure

```bash
python -m opd_sandbox.compare --task reversal \
  --batch_size 64 --num_steps 300 --eval_every 20 --num_seeds 3
```

This writes per-method CSVs, a combined CSV, and `opd_results/reversal_opd_compare.png`.

## Variance microscope

The first appendix experiment measures estimator variance without doing any training. It samples student rollouts from a fixed tiny transformer, computes three OPD-style gradient estimators on the same batches, and reports the variance of a fixed random gradient projection as the horizon grows.

- `sequence_pg`: cumulative sampled-return score estimator.
- `token_pg`: one-step sampled score estimator with $\gamma = 0$ credit.
- `full_vocab_rkl`: exact per-token $\mathrm{KL}(\pi_\text{student} \Vert \pi_\text{teacher})$.

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
