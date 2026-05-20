# opd_sandbox

A small appendix sandbox for on-policy distillation mechanics.

The OPD design space is full of choices that look interchangeable on paper and
behave differently in training: which gradient estimator (sampled vs exact,
sequence vs token), which support to truncate to (student, teacher, or
intersection top-k), when to switch from full vocabulary to truncated, and how
sharp the teacher should be. Each of these has folklore attached, but the
mechanism behind the folklore is hard to test at LLM scale because the runs
are slow and the failure modes get masked by confounds.

This package is the cheap, controlled version: tiny student, full-support
smoothed oracle teacher, short horizons. The teacher is an oracle so the
design-choice question stays clean of teacher-quality confounds. The student
is tiny so the gradient signal stays readable when something looks wrong. Both
choices are about preserving observability rather than benchmark performance.

What I am chasing in each experiment is the failure signature: which
diagnostic moves under each design choice, and what that says about the
mechanism underneath the recipe.

## Methods

| Method | Scope |
| --- | --- |
| `OPDReverseKL` | Full-vocabulary `KL(pi_student || pi_teacher)` on student-sampled prefixes. |
| `OPDTopKReverseKL` | Unnormalized reverse-KL contribution restricted to student, teacher, or intersection top-k support. |
| `OPDPG` | Sampled-token reverse-KL reward `log pi_teacher(a|s) - log pi_old(a|s)` with clipped importance sampling. |

## Tasks

- `reversal`: generate the reverse of a random token string.
- `soft_reversal`: reversal with a graded teacher that prefers the correct
  token, nearby content tokens, and then special tokens.
- `format_answer`: emit an answer tag, modular checksum, then copy suffix
  tokens. This adds a format-vs-answer axis for OPD diagnostics.

## Reading order

The experiments build on each other:

1. `variance_microscope.py`: estimator choice before method choice. This
   isolates why cumulative-return reverse KL grows noisy with horizon while
   per-token OPD stays stable.
2. `topk_stability.py`: the hostile cold-start case for top-k OPD. It shows that
   support restriction can remove the only useful signal when the student and
   teacher are not aligned.
3. `topk_cold_start.py`: partial warmup is not the same as a solved alignment
   problem. Switching too early can erase the teacher's corrective mass outside
   the retained support.
4. `topk_warmup_sweep.py`: the meaning of "top-k" depends on vocabulary size,
   teacher entropy, and student behavior. Excluding one action and excluding
   half the action space are different interventions.
5. `soft_teacher_topk.py`: soft teachers complicate the story. Broader teacher
   support can make truncation less brittle, but it can also dilute the
   exact-token signal used by these diagnostics.

Several losses in this sandbox intentionally use unnormalized top-k mass. That
is a diagnostic choice: it lets omitted teacher mass remain visible rather than
renormalizing it away.

## What I take from these runs

The experiments are small, but they isolate mechanisms that are easy to lose inside a large-model training run. Five inferences I would lead with, then the unpacked version of each.

1. Estimator choice is part of the method, not a downstream implementation detail. The variance microscope shows that cumulative-return sampled reverse KL becomes the dominant source of horizon-dependent noise before optimization has any chance to matter. Per-token OPD does more than save compute in this sandbox; it changes the stability regime the optimizer sees.
2. Top-k is a support-overlap assumption in disguise. Student-top-k OPD fails from a random cold start because the retained support omits the teacher's useful token. The failure mode is unusual: the distribution stays near-uniform and never acquires the teacher's local asymmetry, so there is no sharp mode collapse to diagnose, just a quiet absence of learning.
3. Warmup helps only when it changes behavior. Loss reductions during the full-vocabulary phase do not by themselves create the support overlap that top-k needs; switching too early can erase whatever behavioral asymmetry the warmup produced. The diagnostics that track this are behavioral (top-1 agreement, sampled reward) plus entropy and teacher mass on the selected support, read together. No single overlap number is enough.
4. The width of top-k changes what the experiment is testing. In a nine-action toy task, `k=8` covers all but one action while `k=4` covers less than half. Treating both as the same "top-k" intervention hides the mechanism and produces overbroad conclusions about when truncation works.
5. Teacher entropy is a first-order variable. A broad teacher can be argmax-correct while still being a weak exact-token teacher at fixed compute. Soft support makes top-k less brittle in some regimes and dilutes the corrective signal in others, so the right teacher temperature depends on which reward metric you care about.

Top-k OPD becomes meaningful when three preconditions are in place: the student already occupies a support that contains the teacher's useful tokens, the teacher is sharp enough on that support to produce a meaningful gradient, and the truncated objective still carries the correction the full teacher would have provided. The runs in this appendix probe where each of those preconditions can fail, which is the part that maps the boundary of when top-k is the right tool.

## Quick runs

```bash
python -m opd_sandbox.train --task reversal --method OPDReverseKL \
  --batch_size 64 --num_steps 300 --eval_every 20

python -m opd_sandbox.train --task reversal --method OPDPG \
  --batch_size 64 --num_steps 300 --eval_every 20
```

Those two commands cover the basic exact-versus-sampled comparison on
`reversal`. The package also supports `OPDTopKReverseKL` with
student/teacher/intersection top-k support, the `soft_reversal` and
`format_answer` tasks, and teacher-entropy sweeps. The experiment scripts
under `experiments/` set those up; the analysis docs link below interpret
each run.

## First comparison figure

```bash
python -m opd_sandbox.compare --task reversal \
  --batch_size 64 --num_steps 300 --eval_every 20 --num_seeds 3
```

This writes per-method CSVs, a combined CSV, and
`opd_results/reversal_opd_compare.png`.

## Variance microscope

The first appendix experiment measures estimator variance without doing any
training. It samples student rollouts from a fixed tiny transformer, computes
three OPD-style gradient estimators on the same batches, and reports variance
of a fixed random gradient projection as horizon grows:

- `sequence_pg`: cumulative sampled-return score estimator.
- `token_pg`: one-step sampled score estimator with gamma=0 credit.
- `full_vocab_rkl`: exact per-token `KL(pi_student || pi_teacher)`.

```bash
python -m opd_sandbox.experiments.variance_microscope \
  --horizons 4,8,16,32 --num_batches 40 --num_seeds 3
```

This writes `opd_results/variance_microscope.csv` and
`opd_results/variance_microscope.png`.

The interpreted appendix run lives in
[`analysis/variance_microscope.md`](analysis/variance_microscope.md).

## Top-k stability

The second appendix experiment compares full-vocabulary reverse KL,
student-top-k truncated reverse KL, and sampled-token OPD on the same toy task.

```bash
python -m opd_sandbox.experiments.topk_stability \
  --top_ks 1,2,4 --num_steps 300 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

This writes `topk_stability.csv`, per-variant CSVs, and
`topk_stability.png` in the output directory.

The interpreted cold-start support-overlap stress test lives in
[`analysis/topk_stability.md`](analysis/topk_stability.md).

## Top-k cold start

This follow-up tests whether a full-vocabulary warmup makes top-k truncation
stable:

```bash
python -m opd_sandbox.experiments.topk_cold_start \
  --top_ks 1,2,4 --num_steps 300 --warmup_steps 100 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted run lives in
[`analysis/topk_cold_start.md`](analysis/topk_cold_start.md).

## Top-k warmup sweep

This sweep varies the full-vocabulary warmup length before switching to top-k:

```bash
python -m opd_sandbox.experiments.topk_warmup_sweep \
  --top_ks 4,8 --warmup_steps 0,50,100,150,200,250 \
  --num_steps 300 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted support-threshold result lives in
[`analysis/topk_warmup_sweep.md`](analysis/topk_warmup_sweep.md).

## Soft-teacher top-k support

This follow-up replaces the smoothed one-hot oracle with a graded teacher and
compares student, teacher, and intersection top-k support:

```bash
python -m opd_sandbox.experiments.soft_teacher_topk \
  --top_ks 4 \
  --supports student,teacher,intersection \
  --warmup_steps 0,100,200,250 \
  --num_steps 300 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted support-choice result lives in
[`analysis/soft_teacher_topk.md`](analysis/soft_teacher_topk.md).
