"""CLI for standalone small-model RLM GRPO training."""

from __future__ import annotations

import argparse

from .trainer import RLMGRPOConfig, RLMGRPOTrainer


def parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_config(argv: list[str] | None = None) -> RLMGRPOConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = RLMGRPOConfig()
    parser.add_argument("--model_name", type=str, default=defaults.model_name)
    parser.add_argument("--dataset_name", type=str, default=defaults.dataset_name)
    parser.add_argument("--dataset_split", type=str, default=defaults.dataset_split)
    parser.add_argument("--train_file", type=str, default=defaults.train_file)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max_steps", type=int, default=defaults.max_steps)
    parser.add_argument("--train_batch_size", type=int, default=defaults.train_batch_size)
    parser.add_argument("--group_size", type=int, default=defaults.group_size)
    parser.add_argument("--update_epochs", type=int, default=defaults.update_epochs)
    parser.add_argument(
        "--segment_micro_batch_size",
        type=int,
        default=defaults.segment_micro_batch_size,
    )
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--max_prompt_tokens", type=int, default=defaults.max_prompt_tokens)
    parser.add_argument("--max_children", type=int, default=defaults.max_children)
    parser.add_argument(
        "--max_children_per_call",
        type=int,
        default=defaults.max_children_per_call,
    )
    parser.add_argument("--max_root_turns", type=int, default=defaults.max_root_turns)
    parser.add_argument("--max_child_turns", type=int, default=defaults.max_child_turns)
    parser.add_argument("--repl_timeout", type=int, default=defaults.repl_timeout)
    parser.add_argument(
        "--max_observation_chars",
        type=int,
        default=defaults.max_observation_chars,
    )
    parser.add_argument("--max_plan_tokens", type=int, default=defaults.max_plan_tokens)
    parser.add_argument("--max_child_tokens", type=int, default=defaults.max_child_tokens)
    parser.add_argument(
        "--max_reward_predictions",
        type=int,
        default=defaults.max_reward_predictions,
    )
    parser.add_argument("--reward_mode", type=str, default=defaults.reward_mode)
    parser.add_argument("--judge_model", type=str, default=defaults.judge_model)
    parser.add_argument("--judge_base_url", type=str, default=defaults.judge_base_url)
    parser.add_argument(
        "--judge_api_key_env",
        type=str,
        default=defaults.judge_api_key_env,
    )
    parser.add_argument(
        "--judge_timeout_seconds",
        type=float,
        default=defaults.judge_timeout_seconds,
    )
    parser.add_argument("--judge_max_retries", type=int, default=defaults.judge_max_retries)
    parser.add_argument("--clip_epsilon", type=float, default=defaults.clip_epsilon)
    parser.add_argument("--kl_coef", type=float, default=defaults.kl_coef)
    parser.add_argument("--scale_rewards", type=parse_bool, default=defaults.scale_rewards)
    parser.add_argument(
        "--train_child_trajectories",
        type=parse_bool,
        default=defaults.train_child_trajectories,
    )
    parser.add_argument("--use_peft", type=parse_bool, default=defaults.use_peft)
    parser.add_argument("--load_in_4bit", type=parse_bool, default=defaults.load_in_4bit)
    parser.add_argument("--lora_r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=",".join(defaults.lora_target_modules),
    )
    parser.add_argument("--bf16", type=parse_bool, default=defaults.bf16)
    parser.add_argument(
        "--gradient_checkpointing",
        type=parse_bool,
        default=defaults.gradient_checkpointing,
    )
    parser.add_argument(
        "--trust_remote_code",
        type=parse_bool,
        default=defaults.trust_remote_code,
    )
    parser.add_argument("--require_cuda", type=parse_bool, default=defaults.require_cuda)
    parser.add_argument("--log_every", type=int, default=defaults.log_every)
    parser.add_argument("--save_every", type=int, default=defaults.save_every)
    parser.add_argument("--max_train_examples", type=int, default=defaults.max_train_examples)
    parser.add_argument("--enable_thinking", type=parse_bool, default=defaults.enable_thinking)

    values = vars(parser.parse_args(argv))
    values["lora_target_modules"] = tuple(
        item.strip()
        for item in values["lora_target_modules"].split(",")
        if item.strip()
    )
    return RLMGRPOConfig(**values)


def main(argv: list[str] | None = None):
    config = parse_config(argv)
    trainer = RLMGRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
