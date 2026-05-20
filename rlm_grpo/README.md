# RLM GRPO

A standalone training flow that runs GRPO on small Hugging Face causal LMs (0.5–0.6B) with recursive, tree-of-rollouts sampling.

What I want from this flow is to find out whether a small LM can be trained, under GRPO, to use a Python REPL and recursively call itself to handle sub-tasks of the prompt it was given. The objective is easy to describe and surprisingly intricate to implement, because the rollout shape under recursive self-calls is a tree rather than a sequence. The standard GRPO update was designed for flat rollouts, meaning a group of independent trajectories under the same prompt, and applying it without modification to a tree discards the very structure that recursion was supposed to create. Most of this README is about how to reshape credit propagation so it fits the tree, why the rule I use here is the simplest one that still respects the topology, and what the alternatives would have broken.

## How a training step works

A single prompt produces `group_size` independent rollout trees. Each root rollout interacts with a persistent Python REPL and can, while generating, emit calls to `rlm_query` or `rlm_query_batched`. Those calls spawn child rollouts. The child prompts and the child generation contexts both come from the code the root rollout itself generated, so the shape of the tree (which children exist, how deep they go, what they are asked to do) is decided online by the model rather than pre-specified by the training loop. Two prompts in the same batch can produce trees with very different topologies under the same policy, and a single prompt can produce different topologies across the `group_size` independent rollouts in its group.

That tree-shape variability is what makes flat GRPO an unsafe default. If every decoder step inside the tree is treated as one long trajectory, the recursion structure collapses, and the gradient assigns root-level reward to child segments as if they were continuations of the root's own decoding. If each child is instead trained independently against the root reward, the parent segments that generated the child's prompt are severed from the credit assignment, even though the parent's choice of prompt was what determined whether the child could have succeeded in the first place. Tree-aware GRPO threads between those two failures by keeping the rollout's computation graph intact. Every policy segment in the tree, root or child, retains the path from its own gradient back to the root reward that actually got measured.

The root answer is the only place ground truth lives. The judge, or the `char_f1` scorer in the no-judge configuration, compares the root's predicted evidence against gold evidence and emits a single scalar reward. There is no intermediate-step ground truth at the level of individual REPL calls or individual child outputs, and that absence is deliberate. Building an intermediate-step reward would require either a learned verifier or hand-labeled per-step annotations. A learned verifier would reintroduce the teacher-quality questions this flow is currently holding off, and hand-labeled per-step rewards would commit to a specific account of which intermediate steps should have been good before the model has had a chance to discover its own decomposition strategy. Keeping the reward at the root leaves the decomposition to the model and only checks whether the decomposition worked end to end.

The root reward then has to propagate from the root all the way down through the tree, and the two normalizations the flow uses are where the design choices that matter for training stability live.

- The root advantage is assigned to every policy segment in the tree, root and children alike. If the final answer was right, every segment that contributed to producing it gets credited. If it was wrong, every segment gets debited. This is the simplest credit rule that respects the topology: uniform credit over the segments that produced the rollout. It is the right default here because the alternatives need infrastructure this flow does not have. A per-segment reward signal needs either a learned credit assigner or step-level annotations, and either choice would change the question this flow is asking. The uniform rule keeps the training step legible enough to debug and does not bake in assumptions about which intermediate steps should have been good.

- Child trajectories are first normalized by child count, then split across the generated turns inside each child rollout. The child-count normalization matters more than its description suggests. Without it, a prompt whose tree happens to spawn many children would dominate the gradient relative to a prompt whose tree spawned few, because the same root advantage would be applied to many segments in the first case and few in the second. The disproportion has nothing to do with which prompt was more informative; it is an accident of the tree topology the model itself chose at rollout time. Normalizing by child count makes each prompt contribute equally to the gradient regardless of how many children its tree turned out to have, which is the invariance the trainer needs when the same model can produce trees of very different shapes across prompts in the same batch.

Together, these two normalizations make a training step approximately tree-shape-invariant. The root reward is the only place the gradient gets its sign. The uniform-credit rule propagates that sign through the tree without favoring particular segments. The child-count normalization keeps prompts on equal footing regardless of how branching their rollouts turned out. That invariance is what lets tree-aware GRPO be stable when the same model produces very different rollout topologies across prompts, which is the regime any recursive self-call training has to operate in.

## Install

```bash
pip install -e ".[rlm]"
```

Run from the repository root. The `rlm` extra pulls in `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`, and `httpx` on top of the base sandbox dependencies.

## Run

The default reward uses an OpenAI-compatible judge over the question, the gold evidence, and the predicted evidence:

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
