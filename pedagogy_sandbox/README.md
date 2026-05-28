# pedagogy_sandbox

A toy Pedagogical RL sandbox for privileged rollout acquisition, following the
algorithmic pieces described in
[Pedagogical RL](https://noahziems.com/pedagogical-rl).

Pedagogical RL does not fit the scalar `rl_sandbox.Batch` contract. A training
step owns two policies: a privileged self-teacher conditioned on `(x, c)` and an
unprivileged student conditioned only on `x`. The teacher is trained with GRPO
against a reward scored by the frozen student, then the student assimilates
teacher rollouts with a token-level imitation loss gated by its own surprisal.
Keeping this package separate makes that two-policy contract visible.

## What the local implementation keeps

The scoped implementation is an online alternating toy instantiation that keeps
the method pieces that define the algorithm:

- Teacher-side GRPO over privileged rollouts.
- Product-form pedagogy reward `R(x, c, tau) * G_spike(tau | x)`.
- Spike-aware learnability using the log-sum-exp surprise-gap penalty from the
  blog post.
- Surprisal-gated student imitation, with weights recomputed under the current
  student for each assimilation batch.
- `TeacherRL` as the additive, spike-oblivious ablation: `R - lambda * NLL`,
  followed by vanilla SFT into the student.
- `StudentGRPO` as the purely on-policy baseline.

Out of scope: large-model serving, learned process verifiers, distributed
rollout workers, and claims about the paper's MATH or RiR numbers. The toy task
is a hinted modular-arithmetic sequence problem where the teacher sees the final
answer as privileged context and the student must learn to solve from the input.

## Run

```bash
python -m pedagogy_sandbox.train \
  --method PedagogicalRL \
  --batch_size 96 \
  --group_size 8 \
  --num_steps 300 \
  --eval_every 20 \
  --num_seeds 3
```

Baselines:

```bash
python -m pedagogy_sandbox.train --method StudentGRPO
python -m pedagogy_sandbox.train --method TeacherRL
```

The metrics that decide whether the mechanism is behaving as intended are
`task_reward`, `pedagogy_reward`, `learnability`, `spike_penalty`,
`max_surprise_gap`, `assim_gate_mean`, `teacher_sample_task_reward`, and the
student `test_error`.
