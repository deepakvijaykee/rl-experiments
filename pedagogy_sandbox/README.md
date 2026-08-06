# pedagogy_sandbox

A toy Pedagogical RL sandbox for privileged rollout acquisition, following the algorithmic pieces described in [Pedagogical RL](https://noahziems.com/pedagogical-rl).

Pedagogical RL does not fit the scalar `rl_sandbox.Batch` contract. A training step owns two policies: a privileged self-teacher conditioned on $(x, c)$ and an unprivileged student conditioned only on $x$. The teacher is trained with GRPO against a reward scored by the frozen student, after which the student assimilates teacher rollouts through a token-level imitation loss gated by its own surprisal. Keeping this package separate is what makes that two-policy contract visible.

## The problem it is built around

A purely on-policy student can only receive credit for behavior it already samples. On a task it cannot yet solve, its rollouts are uniformly wrong, the group-centered advantage is zero, and no amount of further sampling produces a direction to move in. This is the cold boundary that [`learnability_sandbox`](../learnability_sandbox/) makes precise, and privileged rollout acquisition is one way across it: let a policy that can see the answer generate the successful trajectories, then hand them to the student.

The difficulty is that correct trajectories are not automatically useful ones. A teacher with the answer in context can produce a solution that is right and still lands so far outside the student's current distribution that imitating it is a large, badly-conditioned jump rather than a lesson. What separates Pedagogical RL from ordinary privileged distillation is that it makes this the teacher's problem: the teacher is scored not on whether its trajectory was correct, but on whether the trajectory was correct *and* the student could absorb it.

## Why the reward is a product

The pedagogy reward is $R(x, c, \tau)\, G_\text{spike}(\tau \mid x)$, where $R$ is task success and $G_\text{spike}$ is a learnability term in $(0, 1]$ computed by scoring the trajectory under the frozen student. Because both factors multiply, both are necessary. A correct trajectory the student cannot follow is discounted toward zero, and an easy trajectory that fails the task is worth exactly zero no matter how familiar it looks.

That is what `TeacherRL` is in the package to test. It uses the additive form $R - \lambda \cdot \mathrm{NLL}$ instead, and the difference is not cosmetic. Under an additive reward a wrong trajectory can buy back score simply by being predictable, so the teacher faces a genuine trade between being right and being easy. The product form forbids that trade by construction, which is the structural claim worth isolating.

The learnability term itself is built from the per-position surprise gap

```math
\text{gap}_t = \max_a \log \pi_\text{student}(a \mid s_t) - \log \pi_\text{student}(a_t \mid s_t)
```

which is zero exactly when the student would already have chosen the teacher's token at that position and grows with how far the teacher's choice sits from the student's own preference. Those per-position gaps are then combined with a log-sum-exp at inverse temperature `beta`, which interpolates between the mean gap as `beta` approaches zero and the maximum gap as `beta` grows. The soft maximum is the point of the design, and it is why the term is spike-aware rather than an average surprisal. A trajectory that is comfortable everywhere except at one step the student has no chance of producing is not a learnable trajectory, but averaging over positions hides exactly that step, since one impossible position among twenty barely moves a mean. Taking something close to a maximum instead lets a single blocking position veto the trajectory, which is what a blocker actually does to the student's ability to follow it.

The toy task is built so that the teacher has room to exercise this. The student sees two input tokens and must emit a short trajectory whose final token is the modular sum, while the teacher sees the same input plus the privileged answer. The scratch tokens before the final answer are unconstrained, so many trajectories reach the same correct answer and differ only in how the teacher got there. Those scratch positions are the degrees of freedom the learnability term selects over, and without them the teacher would have nothing to choose between, collapsing the reward back to task success alone.

## Why the student gate points the way it does

The student side assimilates teacher rollouts with a token-level imitation loss weighted by $\operatorname{sigmoid}(\kappa\,(\log \pi_\text{student}(a_t) - \gamma))$. The weight sits near one on tokens the student already assigns high probability and near zero on tokens it finds surprising. That is the reverse of hard-example mining, and the reversal is easy to read past: the most surprising tokens are precisely the ones this objective ignores.

Read alongside the teacher reward, the two pieces push the same way. The teacher reward selects trajectories with no surprise spikes, and the student gate skips whatever spikes survive that selection, so the update stays inside the neighborhood where the student's current distribution already puts mass. The gate weights are recomputed under the current student for each assimilation batch rather than cached, which they have to be, since the gate is defined relative to a student that moves during assimilation and stale weights would gate against a model that no longer exists.

## What the local implementation keeps

The scoped implementation is an online alternating toy instantiation that keeps the pieces defining the algorithm:

- Teacher-side GRPO over privileged rollouts.
- Product-form pedagogy reward $R(x, c, \tau)\, G_\text{spike}(\tau \mid x)$.
- Spike-aware learnability using the log-sum-exp surprise-gap penalty from the blog post.
- Surprisal-gated student imitation, with weights recomputed under the current student for each assimilation batch.
- `TeacherRL` as the additive, spike-oblivious ablation, $R - \lambda \cdot \mathrm{NLL}$, followed by vanilla SFT into the student.
- `StudentGRPO` as the purely on-policy baseline, which is the arm that has to cross the cold boundary without help.

Out of scope: large-model serving, learned process verifiers, distributed rollout workers, and any claim about the paper's MATH or RiR numbers.

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

## What to read off the run

The metrics that decide whether the mechanism is behaving as intended are `task_reward`, `pedagogy_reward`, `learnability`, `spike_penalty`, `max_surprise_gap`, `assim_gate_mean`, `teacher_sample_task_reward`, and the student `test_error`.

The pairs matter more than the individual numbers. Reading `task_reward` beside `pedagogy_reward` shows how much of the teacher's task success is surviving the learnability discount, and a wide and widening gap means the teacher is solving the task in ways the student cannot use. Reading `spike_penalty` beside `max_surprise_gap` separates a trajectory that is uniformly slightly unfamiliar from one carrying a single blocking position, which is the distinction the soft maximum exists to make. And `assim_gate_mean` reports how much of each teacher trajectory the student is actually willing to imitate, so a gate mean collapsing toward zero says assimilation has quietly stopped even while the teacher's own reward curve continues to look healthy.
