# RLM GRPO

Train 0.5B/0.6B Hugging Face causal LMs with recursive language model rollouts
and a tree-aware GRPO update.

## Training Contract

- Each prompt samples `group_size` independent root rollout trees.
- Root rollouts interact with a persistent Python REPL and can call
  `rlm_query` or `rlm_query_batched` to spawn child rollouts.
- Child prompts and child contexts come from the root's generated REPL code.
- Only the final root answer is rewarded.
- The root advantage is assigned to all generated root and child policy
  segments in the rollout tree.
- Child trajectories are normalized by child count, then split across the
  generated turns inside each child rollout.

## Install

```bash
pip install -r requirements.txt
```

## Run

The default reward mode uses an OpenAI-compatible judge over the question, gold
evidence, and predicted evidence:

```bash
export OPENAI_API_KEY=...
```

Qwen2.5 0.5B:

```bash
python -m rlm_grpo.cli \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir outputs/rlm-qwen25-05b \
  --group_size 4 \
  --train_batch_size 1 \
  --segment_micro_batch_size 1 \
  --max_steps 100
```

Qwen3 0.6B:

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

For deterministic local scoring against gold evidence spans:

```bash
python -m rlm_grpo.cli \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --reward_mode char_f1 \
  --output_dir outputs/rlm-qwen25-05b-char-f1
```

The default dataset is `alphaXiv/multi-paper-synthetic`. To use a local dataset:

```bash
python -m rlm_grpo.cli \
  --train_file /path/to/train.parquet \
  --model_name Qwen/Qwen2.5-0.5B-Instruct
```
