# opd_sandbox

A small appendix sandbox for on-policy distillation mechanics.

This package is intentionally separate from `rl_sandbox/`. The RL sandbox asks
how reward-based policy-gradient updates allocate credit. This OPD sandbox asks
a narrower question: when the teacher is an oracle distribution on
student-visited toy states, which parts of the distillation signal survive the
estimator, support restriction, warmup schedule, and teacher-entropy choices?

The teacher here is a full-support smoothed oracle distribution, not a served
LLM. This is not production OPD and it is not trying to reproduce frontier-scale
numbers. The point is to make a few OPD mechanisms visible in a setting where
the code is short enough to audit and the runs are cheap enough to repeat.

The useful object is the failure signature: which diagnostic moves, which one
does not, and which interpretation is actually supported by the run.

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

## How To Read This Appendix

Read these experiments as mechanism probes, not as a leaderboard. Most runs use
an exact or semi-exact teacher, a deliberately tiny student, and short horizons.
That buys clarity, but it also means the effect sizes and switch thresholds are
not scaling predictions.

The intended reading order is:

1. `variance_microscope.py`: estimator choice before method choice. This
   isolates why cumulative-return reverse KL is noisy and why per-token OPD is
   the stable default in many practical implementations.
2. `topk_stability.py`: the hostile cold-start case for top-k OPD. It shows that
   support restriction can remove the only useful signal when the student and
   teacher are not aligned.
3. `topk_cold_start.py`: partial warmup is not the same as a solved alignment
   problem. Switching too early can erase the teacher's corrective mass outside
   the retained support.
4. `topk_warmup_sweep.py`: the meaning of "top-k" depends on vocabulary size,
   teacher entropy, and student behavior. A mild support cut and a severe
   bottleneck are different interventions.
5. `soft_teacher_topk.py`: soft teachers complicate the story. Broader teacher
   support can make truncation less brittle, but it can also dilute the
   exact-token signal used by these diagnostics.

Several losses in this sandbox intentionally use unnormalized top-k mass. That
is a diagnostic choice: it lets omitted teacher mass remain visible rather than
renormalizing it away.

## What I Take From These Runs

The appendix puts caveats on the clean OPD story. The experiments are small, but
they isolate mechanisms that are easy to lose inside a large-model training run.

1. **Estimator choice is part of the method.** The variance microscope shows
   that cumulative-return sampled reverse KL can become the dominant source of
   horizon-dependent noise before optimization has a chance to matter.
   Per-token OPD is not just a cheaper approximation in this sandbox; it changes
   the stability regime.
2. **Top-k is a support-overlap assumption.** Student-top-k OPD fails from a
   random cold start because the retained support often omits the teacher's
   useful token. The failure is not sharp mode collapse. It is the more subtle
   failure where the distribution stays nearly uniform and never acquires the
   teacher's local asymmetry.
3. **Warmup helps only when it changes behavior, not merely loss.** A short
   full-vocabulary warmup improves top-1 agreement, but switching too early can
   erase that progress. The useful diagnostics are behavioral alignment,
   sampled reward, entropy, and teacher mass on the selected support together.
   No single support-overlap number is enough.
4. **The width of top-k changes the experiment.** In a nine-action toy task,
   `k=8` is almost full vocabulary while `k=4` is a genuine bottleneck. Treating
   both as the same "top-k" intervention hides the mechanism and leads to
   overbroad conclusions.
5. **Teacher entropy is a first-order variable.** A broad teacher can be
   argmax-correct while still being a weak exact-token teacher at a fixed compute
   budget. Soft support makes top-k less brittle in some regimes, but it can
   also dilute the corrective signal.

The practical reading is conservative: these results do not say that top-k OPD
is intrinsically flawed. They say that top-k OPD is only meaningful after asking
what support the student already occupies, how sharp the teacher is on that
support, and whether the retained objective still contains the correction the
full teacher would have provided.

## Quick Runs

```bash
python -m opd_sandbox.train --task reversal --method OPDReverseKL \
  --batch_size 64 --num_steps 300 --eval_every 20

python -m opd_sandbox.train --task reversal --method OPDPG \
  --batch_size 64 --num_steps 300 --eval_every 20
```

Use this package for OPD-specific experiments such as exact reverse KL versus
sampled-token OPD, teacher entropy, top-1 agreement, support overlap, and
KL-zero drift variants.

## First Comparison Figure

```bash
python -m opd_sandbox.compare --task reversal \
  --batch_size 64 --num_steps 300 --eval_every 20 --num_seeds 3
```

This writes per-method CSVs, a combined CSV, and
`opd_results/reversal_opd_compare.png`.

## Variance Microscope

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

## Top-K Stability

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

## Top-K Cold Start

This follow-up tests whether a full-vocabulary warmup makes top-k truncation
stable:

```bash
python -m opd_sandbox.experiments.topk_cold_start \
  --top_ks 1,2,4 --num_steps 300 --warmup_steps 100 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted run lives in
[`analysis/topk_cold_start.md`](analysis/topk_cold_start.md).

## Top-K Warmup Sweep

This sweep varies the full-vocabulary warmup length before switching to top-k:

```bash
python -m opd_sandbox.experiments.topk_warmup_sweep \
  --top_ks 4,8 --warmup_steps 0,50,100,150,200,250 \
  --num_steps 300 --num_seeds 3 \
  --output_dir opd_sandbox/analysis/results
```

The interpreted support-threshold result lives in
[`analysis/topk_warmup_sweep.md`](analysis/topk_warmup_sweep.md).

## Soft-Teacher Top-K Support

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
