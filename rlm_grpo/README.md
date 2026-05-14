# RLM GRPO

A standalone training flow: GRPO on small Hugging Face causal LMs (0.5B–0.6B) with recursive, tree-of-rollouts sampling.

Separate from [`rl_sandbox/`](../rl_sandbox/). The sandbox is for inspecting update rules on toy tasks. This flow is the larger setup, where the model generates code, calls itself through `rlm_query`, and gets scored on the final answer.

## How a training step works

A prompt produces `group_size` independent rollout trees. Each root rollout interacts with a persistent Python REPL and can call `rlm_query` or `rlm_query_batched` to spawn child rollouts. Child prompts and child contexts come from the root's generated REPL code, so the tree shape is decided online.

Only the final root answer earns a reward. That reward then propagates back:

- The root advantage is assigned to every root and child policy segment in the tree.
- Child trajectories are normalized by child count, then split across the generated turns inside each child rollout.

The child-count normalization matters more than it looks. Without it, prompts that happen to spawn many children dominate the gradient on that step.

## Install

```bash
pip install -r requirements.txt
```

## Run

The default reward uses an OpenAI-compatible judge over the question, gold evidence, and predicted evidence:

```bash
export OPENAI_API_KEY=...

python -m rlm_grpo.cli \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/rlm-qwen25-05b \
  --group_size 4 \
  --train_batch_size 1 \
  --segment_micro_batch_size 1 \
  --max_steps 100
```

Qwen3 0.6B with `<think>` disabled:

```bash
python -m rlm_grpo.cli \
  --model_name Qwen/Qwen3-0.6B \
  --enable_thinking false \
  --output_dir outputs/rlm-qwen3-06b \
  --group_size 4 \
  --train_batch_size 1 \
  --segment_micro_batch_size 1 \
  --max_steps 100
```

If you do not want to spend on a judge, score against gold evidence spans directly:

```bash
python -m rlm_grpo.cli \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --reward_mode char_f1 \
  --output_dir outputs/rlm-qwen25-05b-char-f1
```

The default dataset is `alphaXiv/multi-paper-synthetic`. To use a local file:

```bash
python -m rlm_grpo.cli \
  --train_file /path/to/train.parquet \
  --model_name Qwen/Qwen2.5-0.5B-Instruct
```
