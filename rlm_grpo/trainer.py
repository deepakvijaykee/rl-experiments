"""Standalone small-model RLM GRPO trainer.

This HF/PEFT path implements the RLM tree contract directly:

- sample multiple root rollout trees per prompt;
- let root calls induce child rollouts;
- score only the final root answer with the verifier reward;
- compute GRPO advantages within each prompt group;
- propagate each root advantage to its child trajectories with explicit
  child-count normalization.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .data import RLMMultiPaperExample, example_from_row
from .env import CHILD_SYSTEM_PROMPT, ROOT_SYSTEM_PROMPT, RLMReplEnv
from .grpo import (
    CHILD_ROLE, ROOT_ROLE, RLMTrainingSegment, grpo_segment_losses,
    group_relative_advantages, rlm_segment_weights,
)
from .rewards import EvidenceReward, evidence_reward, judge_evidence_reward


@dataclass
class RLMGRPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "alphaXiv/multi-paper-synthetic"
    dataset_split: str = "train"
    train_file: str | None = None
    output_dir: str = "outputs/rlm-grpo"
    seed: int = 0

    max_steps: int = 100
    train_batch_size: int = 1
    group_size: int = 4
    update_epochs: int = 1
    segment_micro_batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    max_prompt_tokens: int = 8192
    max_children: int = 8
    max_children_per_call: int = 4
    max_root_turns: int = 10
    max_child_turns: int = 10
    repl_timeout: int = 30
    max_observation_chars: int = 3000
    max_plan_tokens: int = 384
    max_child_tokens: int = 384
    max_reward_predictions: int = 12
    reward_mode: str = "judge"
    judge_model: str = "gpt-4.1-nano"
    judge_base_url: str = "https://api.openai.com/v1"
    judge_api_key_env: str = "OPENAI_API_KEY"
    judge_timeout_seconds: float = 60.0
    judge_max_retries: int = 5

    clip_epsilon: float = 0.2
    kl_coef: float = 0.01
    scale_rewards: bool = True
    train_child_trajectories: bool = True

    use_peft: bool = True
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ))
    bf16: bool = True
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    require_cuda: bool = True

    log_every: int = 1
    save_every: int = 25
    max_train_examples: int | None = None
    enable_thinking: bool = False

    def validate(self):
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 for GRPO")
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        if self.update_epochs < 1:
            raise ValueError("update_epochs must be >= 1")
        if self.segment_micro_batch_size < 1:
            raise ValueError("segment_micro_batch_size must be >= 1")
        if self.max_children < 1:
            raise ValueError("max_children must be >= 1")
        if self.max_children_per_call < 1:
            raise ValueError("max_children_per_call must be >= 1")
        if self.max_root_turns < 1:
            raise ValueError("max_root_turns must be >= 1")
        if self.max_child_turns < 1:
            raise ValueError("max_child_turns must be >= 1")
        if self.repl_timeout < 1:
            raise ValueError("repl_timeout must be >= 1")
        if self.max_observation_chars < 1:
            raise ValueError("max_observation_chars must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.clip_epsilon < 0:
            raise ValueError("clip_epsilon must be non-negative")
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative")
        if self.reward_mode not in {"judge", "char_f1"}:
            raise ValueError("reward_mode must be 'judge' or 'char_f1'")
        if self.judge_timeout_seconds <= 0:
            raise ValueError("judge_timeout_seconds must be > 0")
        if self.judge_max_retries < 1:
            raise ValueError("judge_max_retries must be >= 1")
        if self.load_in_4bit and not self.use_peft:
            raise ValueError("4-bit training requires use_peft=true")


@dataclass
class GeneratedSegment:
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    old_logprobs: list[float]


@dataclass
class RLMRolloutTree:
    group_id: int
    tree_id: int
    reward: EvidenceReward
    segments: list[RLMTrainingSegment]
    child_count: int
    plan_text: str
    final_text: str


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class RLMGRPOTrainer:
    def __init__(self, config: RLMGRPOConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        self._validate_runtime()
        self._load_model()
        self._load_dataset()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _validate_runtime(self):
        if self.config.require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for this real RLM run. The Codex sandbox may "
                "hide GPUs; run the training script from the host shell or pass "
                "--require_cuda false only for static import/config validation."
            )
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("transformers is required") from exc
        if self.config.use_peft:
            try:
                import peft  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("peft is required for LoRA/QLoRA training") from exc
        if self.config.load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("bitsandbytes is required for load_in_4bit=true") from exc
        if self.config.reward_mode == "judge":
            try:
                import httpx  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("httpx is required for reward_mode='judge'") from exc
            if self.config.judge_api_key_env not in os.environ:
                raise RuntimeError(
                    f"{self.config.judge_api_key_env} must be set for reward_mode='judge'"
                )

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quantization_config = None
        torch_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            trust_remote_code=self.config.trust_remote_code,
        )
        model.config.use_cache = False

        if self.config.use_peft:
            from peft import LoraConfig, TaskType, get_peft_model
            if self.config.load_in_4bit:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=self.config.gradient_checkpointing,
                )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
            )
            model = get_peft_model(model, lora_config)
        elif self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if not self.config.use_peft and self.config.kl_coef > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=self.config.trust_remote_code,
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        self.model = model

    def _load_dataset(self):
        from datasets import load_dataset

        if self.config.train_file:
            suffix = Path(self.config.train_file).suffix
            if suffix == ".jsonl":
                ds = load_dataset("json", data_files=self.config.train_file, split="train")
            elif suffix == ".parquet":
                ds = load_dataset("parquet", data_files=self.config.train_file, split="train")
            else:
                raise ValueError("train_file must be .jsonl or .parquet")
        else:
            ds = load_dataset(self.config.dataset_name, split=self.config.dataset_split)

        if self.config.max_train_examples is not None:
            ds = ds.select(range(min(self.config.max_train_examples, len(ds))))
        if len(ds) < self.config.train_batch_size:
            raise ValueError("dataset is smaller than train_batch_size")
        self.dataset = ds.shuffle(seed=self.config.seed)

    # ------------------------------------------------------------------
    # Tokenization, generation, and logprobs
    # ------------------------------------------------------------------

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
        }
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.config.enable_thinking, **kwargs)
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        ids = encoded[0].tolist()
        if len(ids) > self.config.max_prompt_tokens:
            raise ValueError(
                f"prompt has {len(ids)} tokens, above max_prompt_tokens="
                f"{self.config.max_prompt_tokens}. Reduce context chars or raise the limit."
            )
        return ids

    def _generate_segment(
            self,
            messages: list[dict[str, str]],
            max_new_tokens: int) -> GeneratedSegment:
        prompt_ids = self._encode_messages(messages)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self._model_device())
        attention_mask = torch.ones_like(input_ids)

        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = output[0, input_ids.size(1):].tolist()
        if not completion_ids:
            completion_ids = [self.tokenizer.eos_token_id]
        old_logprobs = self._completion_logprobs(prompt_ids, completion_ids)
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GeneratedSegment(prompt_ids, completion_ids, text, old_logprobs)

    @torch.no_grad()
    def _completion_logprobs(
            self,
            prompt_ids: list[int],
            completion_ids: list[int]) -> list[float]:
        segment = RLMTrainingSegment(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            old_logprobs=[0.0] * len(completion_ids),
            group_id=0,
            tree_id=0,
            role=ROOT_ROLE,
            segment_weight=1.0,
        )
        current, _, mask = self._segment_logprobs([segment], grad=False)
        values = current[mask].detach().float().cpu().tolist()
        if len(values) != len(completion_ids):
            raise RuntimeError("failed to recover completion logprobs")
        return values

    def _segment_batch(self, segments: list[RLMTrainingSegment]):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(s.prompt_ids) + len(s.completion_ids) for s in segments)
        input_rows, label_rows, old_rows, mask_rows = [], [], [], []
        for segment in segments:
            full = segment.prompt_ids + segment.completion_ids
            prompt_len = len(segment.prompt_ids)
            labels = [-100] * prompt_len + segment.completion_ids
            pad = max_len - len(full)
            full = full + [pad_id] * pad
            labels = labels + [-100] * pad

            old = [0.0] * (max_len - 1)
            mask = [False] * (max_len - 1)
            for j, old_logp in enumerate(segment.old_logprobs):
                shifted_idx = prompt_len + j - 1
                if shifted_idx < 0:
                    raise ValueError("prompt must contain at least one token")
                old[shifted_idx] = old_logp
                mask[shifted_idx] = True

            input_rows.append(full)
            label_rows.append(labels)
            old_rows.append(old)
            mask_rows.append(mask)

        device = self._model_device()
        return (
            torch.tensor(input_rows, dtype=torch.long, device=device),
            torch.tensor(label_rows, dtype=torch.long, device=device),
            torch.tensor(old_rows, dtype=torch.float, device=device),
            torch.tensor(mask_rows, dtype=torch.bool, device=device),
        )

    def _segment_logprobs(
            self,
            segments: list[RLMTrainingSegment],
            grad: bool = True,
            ref: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids, labels, old_logprobs, mask = self._segment_batch(segments)
        model = self.ref_model if ref and self.ref_model is not None else self.model
        adapter_ctx = nullcontext()
        if ref and self.config.use_peft:
            adapter_ctx = model.disable_adapter()
        grad_ctx = nullcontext() if grad else torch.no_grad()
        with adapter_ctx, grad_ctx:
            logits = model(input_ids=input_ids).logits.float()
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].clone()
            gather_labels = shift_labels.clamp(min=0)
            logprobs = F.log_softmax(shift_logits, dim=-1)
            selected = logprobs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
            selected = torch.where(shift_labels != -100, selected, torch.zeros_like(selected))
        return selected, old_logprobs, mask

    # ------------------------------------------------------------------
    # RLM rollouts
    # ------------------------------------------------------------------

    def _rollout_tree(
            self,
            example: RLMMultiPaperExample,
            group_id: int,
            tree_id: int) -> RLMRolloutTree:
        context = {
            paper.paper_id: paper.context()
            for paper in example.papers
        }
        child_rollouts: list[list[GeneratedSegment]] = []

        def run_children(prompts, context_list=None):
            if type(prompts) is not list:
                raise ValueError("rlm_query_batched expects prompts to be a list")
            if context_list is None:
                context_list = [context] * len(prompts)
            if type(context_list) is not list:
                raise ValueError("rlm_query_batched context_list must be a list")
            if len(prompts) != len(context_list):
                raise ValueError("prompts and context_list must have equal length")
            for prompt in prompts:
                if type(prompt) is not str:
                    raise ValueError("rlm_query_batched prompts must be strings")
            if len(prompts) > self.config.max_children_per_call:
                raise ValueError(
                    "rlm_query_batched received "
                    f"{len(prompts)} prompts, above max_children_per_call="
                    f"{self.config.max_children_per_call}"
                )
            if root_env.child_count + len(prompts) > self.config.max_children:
                raise ValueError(
                    f"child call budget exceeded: max_children={self.config.max_children}")

            outputs = []
            for prompt, child_context in zip(prompts, context_list):
                if child_context is None:
                    child_context = context
                child_env = RLMReplEnv(
                    system_prompt=CHILD_SYSTEM_PROMPT,
                    user_prompt=f"Query:\n{prompt}",
                    initial_state={"context": child_context},
                    generate_fn=self._generate_segment,
                    max_new_tokens=self.config.max_child_tokens,
                    max_turns=self.config.max_child_turns,
                    max_observation_chars=self.config.max_observation_chars,
                    repl_timeout=self.config.repl_timeout,
                )
                child_result = child_env.run()
                child_rollouts.append(child_result.segments)
                root_env.child_count += 1
                outputs.append(self._parse_child_result(child_result.final_answer))
            return outputs

        def run_child(prompt, context=None):
            if context is None:
                return run_children([prompt])[0]
            return run_children([prompt], context_list=[context])[0]

        root_env = RLMReplEnv(
            system_prompt=ROOT_SYSTEM_PROMPT,
            user_prompt=f"Query:\n{example.question}",
            initial_state={"context": context},
            generate_fn=self._generate_segment,
            max_new_tokens=self.config.max_plan_tokens,
            max_turns=self.config.max_root_turns,
            max_observation_chars=self.config.max_observation_chars,
            repl_timeout=self.config.repl_timeout,
            rlm_query_fn=run_child,
            rlm_query_batched_fn=run_children,
        )

        root_result = root_env.run()
        reward = self._score_reward(
            example,
            root_result.final_answer,
        )

        root_weight, child_weights = rlm_segment_weights(
            root_segment_count=len(root_result.segments),
            child_rollout_segment_counts=[
                len(child_segments) for child_segments in child_rollouts
            ],
            train_child_trajectories=self.config.train_child_trajectories,
        )
        segments = [
            RLMTrainingSegment(
                prompt_ids=part.prompt_ids,
                completion_ids=part.completion_ids,
                old_logprobs=part.old_logprobs,
                group_id=group_id,
                tree_id=tree_id,
                role=ROOT_ROLE,
                segment_weight=root_weight,
            )
            for part in root_result.segments
        ]
        flat_children = [
            child
            for child_segments in child_rollouts
            for child in child_segments
        ]
        for child, child_weight in zip(flat_children, child_weights):
            segments.append(RLMTrainingSegment(
                prompt_ids=child.prompt_ids,
                completion_ids=child.completion_ids,
                old_logprobs=child.old_logprobs,
                group_id=group_id,
                tree_id=tree_id,
                role=CHILD_ROLE,
                segment_weight=child_weight,
            ))

        return RLMRolloutTree(
            group_id=group_id,
            tree_id=tree_id,
            reward=reward,
            segments=segments,
            child_count=root_env.child_count,
            plan_text=root_result.segments[0].text,
            final_text=root_result.final_answer,
        )

    @staticmethod
    def _parse_child_result(result: str) -> Any:
        try:
            return ast.literal_eval(result)
        except (SyntaxError, ValueError):
            return result

    def _score_reward(
            self,
            example: RLMMultiPaperExample,
            final_answer: str) -> EvidenceReward:
        if self.config.reward_mode == "char_f1":
            return evidence_reward(
                final_answer,
                example.reward_spec,
                example.full_context(),
                max_predictions=self.config.max_reward_predictions,
            )
        return judge_evidence_reward(
            final_answer,
            question=example.question,
            reward_spec=example.reward_spec,
            model=self.config.judge_model,
            base_url=self.config.judge_base_url,
            api_key_env=self.config.judge_api_key_env,
            timeout_seconds=self.config.judge_timeout_seconds,
            max_retries=self.config.judge_max_retries,
        )

    def _collect_rollouts(self, rows: list[dict[str, Any]]) -> list[RLMRolloutTree]:
        trees: list[RLMRolloutTree] = []
        tree_id = 0
        for group_id, row in enumerate(rows):
            example = example_from_row(row)
            for _ in range(self.config.group_size):
                trees.append(self._rollout_tree(example, group_id, tree_id))
                tree_id += 1
        return trees

    @staticmethod
    def _assign_advantages(trees: list[RLMRolloutTree], scale_rewards: bool):
        rewards = torch.tensor([tree.reward.score for tree in trees], dtype=torch.float)
        group_ids = torch.tensor([tree.group_id for tree in trees], dtype=torch.long)
        advantages = group_relative_advantages(
            rewards, group_ids, scale_rewards=scale_rewards)
        for tree, advantage in zip(trees, advantages.tolist()):
            for segment in tree.segments:
                segment.advantage = float(advantage)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _segment_loss(
            self,
            segments: list[RLMTrainingSegment],
            num_trees: int) -> tuple[torch.Tensor, dict[str, float]]:
        current, old, mask = self._segment_logprobs(segments, grad=True)
        ref_logprobs = None
        if self.config.kl_coef > 0:
            ref_logprobs, _, _ = self._segment_logprobs(segments, grad=False, ref=True)
        advantages = torch.tensor(
            [s.advantage for s in segments],
            dtype=torch.float,
            device=current.device,
        )
        segment_losses, metrics = grpo_segment_losses(
            current_logprobs=current,
            old_logprobs=old,
            advantages=advantages,
            mask=mask,
            clip_epsilon=self.config.clip_epsilon,
            ref_logprobs=ref_logprobs,
            kl_coef=self.config.kl_coef,
        )
        weights = torch.tensor(
            [s.segment_weight for s in segments],
            dtype=torch.float,
            device=current.device,
        )
        loss = (segment_losses * weights).sum() / max(num_trees, 1)
        metrics["segment_loss"] = segment_losses.mean().item()
        return loss, metrics

    def _update(self, segments: list[RLMTrainingSegment], num_trees: int) -> dict[str, float]:
        self.model.train()
        metrics_accum: list[dict[str, float]] = []
        for _ in range(self.config.update_epochs):
            order = list(range(len(segments)))
            self.rng.shuffle(order)
            self.optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(order), self.config.segment_micro_batch_size):
                idx = order[start:start + self.config.segment_micro_batch_size]
                micro_segments = [segments[i] for i in idx]
                loss, metrics = self._segment_loss(micro_segments, num_trees)
                loss.backward()
                metrics["loss"] = loss.item()
                metrics_accum.append(metrics)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        out = {}
        for key in metrics_accum[0]:
            out[key] = _mean(m[key] for m in metrics_accum if key in m)
        return out

    def train(self):
        if self.model is None:
            self.setup()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_len = len(self.dataset)
        cursor = 0

        for step in range(1, self.config.max_steps + 1):
            rows = []
            for _ in range(self.config.train_batch_size):
                rows.append(self.dataset[cursor % dataset_len])
                cursor += 1

            t0 = time.time()
            trees = self._collect_rollouts(rows)
            self._assign_advantages(trees, self.config.scale_rewards)
            segments = [segment for tree in trees for segment in tree.segments]
            metrics = self._update(segments, num_trees=len(trees))

            if step % self.config.log_every == 0:
                rewards = [tree.reward.score for tree in trees]
                child_counts = [tree.child_count for tree in trees]
                root_segments = [s for s in segments if s.role == ROOT_ROLE]
                child_segments = [s for s in segments if s.role == CHILD_ROLE]
                print(
                    f"step={step} "
                    f"reward={_mean(rewards):.4f} "
                    f"reward_max={max(rewards):.4f} "
                    f"children={_mean(child_counts):.2f} "
                    f"segments={len(segments)} "
                    f"root_segments={len(root_segments)} "
                    f"child_segments={len(child_segments)} "
                    f"loss={metrics['loss']:.4f} "
                    f"ratio={metrics['ratio_mean']:.4f} "
                    f"kl={metrics['kl_mean']:.4f} "
                    f"secs={time.time() - t0:.1f}",
                    flush=True,
                )

            if step % self.config.save_every == 0:
                self.save(output_dir / f"checkpoint-{step}")

        self.save(output_dir / "final")

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(path / "rlm_grpo_config.json", "w", encoding="utf-8") as f:
            import json
            json.dump(dataclasses.asdict(self.config), f, indent=2)
