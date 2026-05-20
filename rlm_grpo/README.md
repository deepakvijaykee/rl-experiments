# RLM GRPO

A standalone training flow: GRPO on small Hugging Face causal LMs (0.5B-0.6B) with recursive, tree-of-rollouts sampling.

The question this flow tests: can a small LM be trained, under GRPO, to use a Python REPL and recursively call itself to handle sub-tasks? The interesting design problem is what to do with the tree of rollouts that recursive calls produce. The standard GRPO update assumes a flat rollout, not a tree; the bulk of this README is about how to fix that.

## How a training step works

A prompt produces `group_size` independent rollout trees. Each root rollout interacts with a persistent Python REPL and can call `rlm_query` or `rlm_query_batched` to spawn child rollouts. The child prompts and child contexts come from the root's generated REPL code, so the tree shape is decided online by the model itself.

When a model recursively calls itself, the natural rollout topology is a tree, and flattening it discards the structure. Treating all decoder steps as one long trajectory collapses the recursion. Training each child independently severs the root reward from the parent segments that produced the child's prompt. Tree-aware GRPO keeps the rollout's computation graph intact, which is the design choice the setup is built around.

Only the final root answer earns a reward. That is the only place ground truth lives: the judge or `char_f1` scorer compares predicted evidence against gold evidence, and there is no intermediate-step ground truth unless we stand up a learned verifier. A learned verifier would be a separate project, and it would reintroduce the teacher-quality question this flow is currently holding off. Keeping the reward at the root is also conservative: it avoids baking in assumptions about which intermediate steps should have been good.

The root reward propagates back through the tree:

- The root advantage is assigned to every policy segment in the tree, root and children alike. If the final answer was right, every segment that contributed to producing it gets credited; if it was wrong, every segment gets debited. That is the simplest defensible credit rule for this topology. Anything fancier (per-segment reward, learned credit) needs infrastructure I have not built here; the simple rule keeps the training step readable and debuggable.

- Child trajectories are normalized by child count, then split across the generated turns inside each child rollout. The child-count normalization matters more than it looks. Without it, prompts that happen to spawn many children dominate the gradient on that step, because the same root advantage gets applied to many segments. Normalizing by child count makes each prompt contribute equally to the gradient regardless of how many children its tree turned out to have. That matters because the model decides the tree shape online, and tree shapes vary across prompts.

Together these two normalizations make a training step approximately tree-shape-invariant. That is what lets tree-aware GRPO be stable when the same model produces very different rollout topologies across prompts.

## Install

```bash
pip install -e ".[rlm]"
```

Run from the repository root. The `rlm` extra pulls in `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`, and `httpx` on top of the base sandbox dependencies.

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
