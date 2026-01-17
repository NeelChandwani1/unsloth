#!/usr/bin/env python3
"""gemma3_270m_unsloth_grpo.py

Group Relative Policy Optimization (GRPO) training for Gemma 3 270M using
Unsloth + Hugging Face TRL.

This script is a *local* (non-notebook) training entrypoint. It is designed
for an environment where:
  - CUDA is available
  - Unsloth is installed
  - TRL is installed (with GRPOTrainer / GRPOConfig)

It is an RL (online) training loop:
  1) Sample a batch of prompts from the dataset.
  2) Generate multiple completions per prompt ("group").
  3) Score each completion with reward function(s).
  4) Update the policy using the GRPO objective.

Dataset expectations
--------------------
This script defaults to the public dataset used in the SFT script:
    "Thytu/ChessInstruct"

The default mapper expects the following columns to exist in the raw dataset:
  - task            (system prompt)
  - input           (user message)
  - expected_output (reference/target answer)

It converts each sample to the TRL conversational GRPO format:
  - prompt:  list[dict]  (messages with {"role": ..., "content": ...})
  - answer:  str         (reference answer used by reward functions)

Reward functions
----------------
GRPO relies on *relative* comparisons among the group of completions sampled
for the same prompt. A purely binary reward (0/1) can lead to long stretches
where all completions are "wrong" and receive identical reward, producing
near-zero learning signal.

The default reward here is therefore *shaped*:
  - Exact match / substring bonuses for the reference answer
  - Similarity-based shaping (difflib ratio)
  - Non-empty completion reward

You should replace the reward functions with domain-specific verifiers for best
results.

How to run
----------
Single GPU:
    python gemma3_270m_unsloth_grpo.py

Recommended (Accelerate launcher; also works for multi-GPU):
    accelerate launch gemma3_270m_unsloth_grpo.py

Common knobs:
    --max_steps 300
    --num_generations 4
    --per_device_train_batch_size 4

Notes on GRPO batching:
-----------------------
In TRL, the *effective* training batch size must typically be divisible by
num_generations. Recent TRL versions may enforce this.

If you see an error about divisibility:
  - Increase per_device_train_batch_size to a multiple of num_generations, OR
  - Decrease num_generations.

Saving
------
By default, this saves the resulting (PEFT) adapter to --save_dir.

"""

from __future__ import annotations

import argparse
import difflib
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)


# ----------------------------
# Configuration / CLI
# ----------------------------


@dataclass(frozen=True)
class GRPOTrainConfig:
    # Model / tokenizer
    model_name: str = "unsloth/gemma-3-270m-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    full_finetuning: bool = False
    hf_token: Optional[str] = None

    # LoRA (PEFT)
    lora_r: int = 128
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Any = None

    # Chat template
    chat_template: str = "gemma3"

    # Dataset
    dataset_name: str = "Thytu/ChessInstruct"
    dataset_split: str = "train[:10000]"
    dataset_task_field: str = "task"
    dataset_input_field: str = "input"
    dataset_output_field: str = "expected_output"

    # GRPO training args
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 0.001
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    save_steps: int = 100
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"

    # GRPO / sampling hyperparameters
    num_generations: int = 2
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0

    # Sequence lengths
    # If not provided, we estimate a safe max_prompt_length from the dataset.
    max_prompt_length: Optional[int] = None
    # If not provided, we set max_completion_length = max_seq_length - max_prompt_length
    # (mirroring common notebook patterns).
    max_completion_length: Optional[int] = None

    # Core GRPO objective knobs
    beta: float = 0.0
    num_iterations: int = 1
    epsilon: float = 0.2
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    mask_truncated_completions: bool = False

    # Logging
    log_completions: bool = False
    num_completions_to_print: Optional[int] = None

    # Prompt length estimation (only used if max_prompt_length is None)
    prompt_length_sample_size: int = 256
    prompt_length_margin: int = 8

    # Saving
    save_dir: str = "gemma-3-grpo"

    # Display
    show_examples: bool = True


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _require_environment() -> None:
    if sys.version_info < (3, 9):
        raise RuntimeError("Python >= 3.9 is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run GRPO training.")


def _log_library_versions() -> None:
    """Best-effort version logging (never fatal)."""

    try:
        import transformers
        import trl
        import datasets

        LOGGER.info("transformers=%s", transformers.__version__)
        LOGGER.info("trl=%s", trl.__version__)
        LOGGER.info("datasets=%s", datasets.__version__)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Version logging skipped: %s", exc)

    try:
        import unsloth

        LOGGER.info("unsloth=%s", getattr(unsloth, "__version__", "(unknown)"))
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("Unsloth version logging skipped: %s", exc)


def _get_world_size() -> int:
    """Return the world size if launched via accelerate/torchrun, else 1."""

    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        val = os.environ.get(key)
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                continue
    return 1


def parse_args(argv: Optional[List[str]] = None) -> Tuple[GRPOTrainConfig, int]:
    parser = argparse.ArgumentParser(
        description=(
            "Train unsloth/gemma-3-270m-it with TRL GRPOTrainer (Group Relative Policy Optimization) "
            "using an Unsloth-accelerated model."  # noqa: E501
        )
    )

    # Model
    parser.add_argument("--model_name", default=GRPOTrainConfig.model_name)
    parser.add_argument("--max_seq_length", type=int, default=GRPOTrainConfig.max_seq_length)
    parser.add_argument("--load_in_4bit", action="store_true", default=GRPOTrainConfig.load_in_4bit)
    parser.add_argument("--load_in_8bit", action="store_true", default=GRPOTrainConfig.load_in_8bit)
    parser.add_argument("--full_finetuning", action="store_true", default=GRPOTrainConfig.full_finetuning)
    parser.add_argument("--hf_token", default=None)

    # Dataset
    parser.add_argument("--dataset_name", default=GRPOTrainConfig.dataset_name)
    parser.add_argument("--dataset_split", default=GRPOTrainConfig.dataset_split)
    parser.add_argument("--dataset_task_field", default=GRPOTrainConfig.dataset_task_field)
    parser.add_argument("--dataset_input_field", default=GRPOTrainConfig.dataset_input_field)
    parser.add_argument("--dataset_output_field", default=GRPOTrainConfig.dataset_output_field)

    # Training
    parser.add_argument("--per_device_train_batch_size", type=int, default=GRPOTrainConfig.per_device_train_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRPOTrainConfig.gradient_accumulation_steps)
    parser.add_argument("--max_steps", type=int, default=GRPOTrainConfig.max_steps)
    parser.add_argument("--learning_rate", type=float, default=GRPOTrainConfig.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=GRPOTrainConfig.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=GRPOTrainConfig.warmup_ratio)
    parser.add_argument("--lr_scheduler_type", default=GRPOTrainConfig.lr_scheduler_type)
    parser.add_argument("--optim", default=GRPOTrainConfig.optim)
    parser.add_argument("--logging_steps", type=int, default=GRPOTrainConfig.logging_steps)
    parser.add_argument("--save_steps", type=int, default=GRPOTrainConfig.save_steps)
    parser.add_argument("--seed", type=int, default=GRPOTrainConfig.seed)
    parser.add_argument("--output_dir", default=GRPOTrainConfig.output_dir)
    parser.add_argument("--report_to", default=GRPOTrainConfig.report_to)

    # GRPO sampling
    parser.add_argument("--num_generations", type=int, default=GRPOTrainConfig.num_generations)
    parser.add_argument("--temperature", type=float, default=GRPOTrainConfig.temperature)
    parser.add_argument("--top_p", type=float, default=GRPOTrainConfig.top_p)
    parser.add_argument("--top_k", type=int, default=GRPOTrainConfig.top_k)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=GRPOTrainConfig.repetition_penalty)

    # Lengths
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--prompt_length_sample_size", type=int, default=GRPOTrainConfig.prompt_length_sample_size)
    parser.add_argument("--prompt_length_margin", type=int, default=GRPOTrainConfig.prompt_length_margin)

    # GRPO objective knobs
    parser.add_argument("--beta", type=float, default=GRPOTrainConfig.beta)
    parser.add_argument("--num_iterations", type=int, default=GRPOTrainConfig.num_iterations)
    parser.add_argument("--epsilon", type=float, default=GRPOTrainConfig.epsilon)
    parser.add_argument("--loss_type", default=GRPOTrainConfig.loss_type)
    parser.add_argument("--scale_rewards", default=GRPOTrainConfig.scale_rewards)
    parser.add_argument(
        "--mask_truncated_completions",
        action=argparse.BooleanOptionalAction,
        default=GRPOTrainConfig.mask_truncated_completions,
    )

    # Logging extras
    parser.add_argument(
        "--log_completions",
        action=argparse.BooleanOptionalAction,
        default=GRPOTrainConfig.log_completions,
    )
    parser.add_argument("--num_completions_to_print", type=int, default=None)

    # Saving / output
    parser.add_argument("--save_dir", default=GRPOTrainConfig.save_dir)
    parser.add_argument(
        "--show_examples",
        action=argparse.BooleanOptionalAction,
        default=GRPOTrainConfig.show_examples,
    )

    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args(argv)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    cfg = GRPOTrainConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
        hf_token=hf_token,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_task_field=args.dataset_task_field,
        dataset_input_field=args.dataset_input_field,
        dataset_output_field=args.dataset_output_field,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        prompt_length_sample_size=args.prompt_length_sample_size,
        prompt_length_margin=args.prompt_length_margin,
        beta=args.beta,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=args.mask_truncated_completions,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        save_dir=args.save_dir,
        show_examples=args.show_examples,
    )

    return cfg, args.verbose


# ----------------------------
# Dataset preparation
# ----------------------------


def _validate_dataset_columns(dataset: Any, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in getattr(dataset, "column_names", [])]
    if missing:
        raise KeyError(
            "Dataset is missing required columns: "
            f"{missing}. Available columns: {getattr(dataset, 'column_names', None)}"
        )


def convert_to_grpo_conversational(
    example: Dict[str, Any],
    *,
    task_field: str,
    input_field: str,
    output_field: str,
) -> Dict[str, Any]:
    """Convert a single dataset example into TRL's conversational GRPO format."""

    return {
        "prompt": [
            {"role": "system", "content": str(example[task_field])},
            {"role": "user", "content": str(example[input_field])},
        ],
        "answer": str(example[output_field]),
    }


def estimate_max_prompt_length(
    dataset: Any,
    tokenizer: Any,
    *,
    sample_size: int,
    margin: int,
    max_seq_length: int,
) -> int:
    """Estimate a safe `max_prompt_length` for GRPO.

    We measure the tokenized length of the *formatted* prompt (including the
    generation prompt) for up to `sample_size` samples and take the maximum.

    The returned value is clamped to `[1, max_seq_length - 1]`.
    """

    n = min(int(sample_size), len(dataset))
    if n <= 0:
        return max(1, min(512, max_seq_length - 1))

    max_len = 0
    for i in range(n):
        messages = dataset[i]["prompt"]
        # tokenize=True returns token ids directly
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        max_len = max(max_len, len(ids))

    max_len = max_len + int(margin)
    return max(1, min(int(max_len), int(max_seq_length) - 1))


def resolve_prompt_and_completion_lengths(
    cfg: GRPOTrainConfig,
    dataset: Any,
    tokenizer: Any,
) -> Tuple[int, int]:
    """Resolve max_prompt_length and max_completion_length.

    Strategy:
      - If cfg.max_prompt_length is provided, use it.
      - Else estimate it from the dataset.

      - If cfg.max_completion_length is provided, use it.
      - Else set it to (cfg.max_seq_length - max_prompt_length).

    Finally, ensure both are at least 1 and that their sum does not exceed
    cfg.max_seq_length.
    """

    max_prompt_length = cfg.max_prompt_length
    if max_prompt_length is None:
        max_prompt_length = estimate_max_prompt_length(
            dataset,
            tokenizer,
            sample_size=cfg.prompt_length_sample_size,
            margin=cfg.prompt_length_margin,
            max_seq_length=cfg.max_seq_length,
        )

    max_prompt_length = max(1, min(int(max_prompt_length), cfg.max_seq_length - 1))

    max_completion_length = cfg.max_completion_length
    if max_completion_length is None:
        max_completion_length = cfg.max_seq_length - max_prompt_length

    max_completion_length = max(1, int(max_completion_length))

    # Ensure the total context doesn't exceed max_seq_length.
    if max_prompt_length + max_completion_length > cfg.max_seq_length:
        max_completion_length = max(1, cfg.max_seq_length - max_prompt_length)

    return int(max_prompt_length), int(max_completion_length)


# ----------------------------
# Reward functions
# ----------------------------


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


def _completion_to_text(completion: Any) -> str:
    """Coerce TRL completion formats into plain text.

    TRL passes different completion structures depending on dataset format:
      - Standard format: str
      - Conversational format: list[dict] where each dict has a "content" key

    This helper is defensive and works across both.
    """

    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        # Conversational: typically list[{role, content}] (often length 1)
        parts: List[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _extract_answer_text(text: str) -> str:
    """Extract a "final answer" from common structured formats."""

    if not text:
        return ""

    m = _ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()

    # Common in some math datasets: "#### <answer>"
    if "####" in text:
        return text.split("####", 1)[-1].strip()

    # Heuristic: if the model writes "Answer: ..." take the tail.
    lowered = text.lower()
    if "answer:" in lowered:
        idx = lowered.rfind("answer:")
        return text[idx + len("answer:") :].strip()

    return text.strip()


def _normalize_for_match(text: str) -> str:
    """Normalize text for loose matching.

    We intentionally keep alphanumerics and a small subset of symbols that are
    common in chess moves / answers (e.g., '+', '#', '=', '-', 'x').
    """

    text = _extract_answer_text(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing punctuation while retaining core symbols.
    text = text.strip(" \t\n\r\f\v\"'`.,;:()[]{}")

    # Collapse spaces to simplify substring match.
    text = text.replace(" ", "")
    return text


def reward_nonempty(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """Small reward for producing a non-empty completion."""

    rewards: List[float] = []
    for c in completions:
        txt = _completion_to_text(c).strip()
        rewards.append(1.0 if txt else 0.0)
    return rewards


def reward_shaped_reference_match(
    completions: Sequence[Any],
    answer: Sequence[str],
    **kwargs: Any,
) -> List[float]:
    """Reward based on similarity to a reference answer.

    Returns a shaped score:
      - Empty completion: -1.0
      - Otherwise: difflib ratio in [0, 1] + bonus

    Bonus terms:
      - +1.0 if exact match after normalization
      - +0.5 if the (normalized) reference answer is a substring of the
        (normalized) completion and the reference is short (helps for cases
        where the model writes explanations around a short answer).

    The final reward is typically in [-1, 2].
    """

    rewards: List[float] = []

    for c, a in zip(completions, answer):
        comp = _completion_to_text(c)
        if not comp or not comp.strip():
            rewards.append(-1.0)
            continue

        comp_norm = _normalize_for_match(comp)
        ans_norm = _normalize_for_match(str(a))

        if not ans_norm:
            # No reference available; fall back to a weak non-empty reward.
            rewards.append(0.0)
            continue

        ratio = difflib.SequenceMatcher(None, comp_norm, ans_norm).ratio()

        bonus = 0.0
        if comp_norm == ans_norm:
            bonus = 1.0
        elif len(ans_norm) <= 16 and ans_norm in comp_norm:
            bonus = 0.5

        rewards.append(float(ratio + bonus))

    return rewards


def reward_special_token_penalty(completions: Sequence[Any], **kwargs: Any) -> List[float]:
    """Penalize leaking template/control tokens into the visible completion."""

    bad_markers = (
        "<start_of_turn>",
        "<end_of_turn>",
        "<bos>",
        "</s>",
        "<eos>",
    )

    rewards: List[float] = []
    for c in completions:
        txt = _completion_to_text(c)
        lowered = txt.lower()
        penalty = 0.0
        for m in bad_markers:
            if m in lowered:
                penalty -= 0.25
        rewards.append(penalty)
    return rewards


# ----------------------------
# Main
# ----------------------------


def _filter_kwargs_for_callable(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a kwargs dict to only include parameters supported by `callable_obj`.

    This keeps the script compatible across multiple TRL versions where GRPOConfig
    or GRPOTrainer may add/remove parameters.
    """

    import inspect

    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    supported = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in supported}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass


def _load_unsloth_model(cfg: GRPOTrainConfig):
    """Load model + tokenizer via Unsloth.

    We support both `FastModel` and `FastLanguageModel` (depending on the
    installed Unsloth version).
    """

    try:
        from unsloth import FastModel as FastUnslothModel
    except Exception:  # pragma: no cover
        from unsloth import FastLanguageModel as FastUnslothModel

    model, tokenizer = FastUnslothModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        full_finetuning=cfg.full_finetuning,
        token=cfg.hf_token,
    )

    model = FastUnslothModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=list(cfg.target_modules),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.random_state,
        use_rslora=cfg.use_rslora,
        loftq_config=cfg.loftq_config,
    )

    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)

    # TRL GRPOTrainer requires left padding.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None:
        # TRL will fall back to eos_token, but we make it explicit.
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _prepare_dataset(cfg: GRPOTrainConfig, tokenizer: Any):
    from datasets import load_dataset

    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)

    _validate_dataset_columns(
        dataset,
        required=[cfg.dataset_task_field, cfg.dataset_input_field, cfg.dataset_output_field],
    )

    # Map to the required GRPO fields and drop the original columns.
    original_columns = list(dataset.column_names)

    dataset = dataset.map(
        lambda ex: convert_to_grpo_conversational(
            ex,
            task_field=cfg.dataset_task_field,
            input_field=cfg.dataset_input_field,
            output_field=cfg.dataset_output_field,
        ),
        remove_columns=original_columns,
    )

    if cfg.show_examples and len(dataset) > 0:
        LOGGER.info("Example prompt: %s", dataset[0]["prompt"])
        LOGGER.info("Example answer: %s", dataset[0]["answer"])

        # Also log the tokenized prompt length for sanity.
        ids = tokenizer.apply_chat_template(dataset[0]["prompt"], tokenize=True, add_generation_prompt=True)
        LOGGER.info("Example prompt token length (with generation prompt): %d", len(ids))

    return dataset


def _maybe_warn_about_grpo_batching(cfg: GRPOTrainConfig) -> None:
    if cfg.num_generations < 2:
        raise ValueError("For GRPO, --num_generations must be >= 2 to compute group-relative advantages.")

    world_size = _get_world_size()
    effective_bs = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size

    # Some TRL versions enforce this; others do not. We warn either way.
    if effective_bs % cfg.num_generations != 0:
        LOGGER.warning(
            "GRPO batching: effective batch size (%d = world_size=%d * per_device_train_batch_size=%d * "
            "grad_accum=%d) is not divisible by num_generations=%d. "
            "If TRL raises an error, increase per_device_train_batch_size to a multiple of num_generations "
            "or reduce num_generations.",
            effective_bs,
            world_size,
            cfg.per_device_train_batch_size,
            cfg.gradient_accumulation_steps,
            cfg.num_generations,
        )


def main(argv: Optional[List[str]] = None) -> int:
    cfg, verbosity = parse_args(argv)
    _configure_logging(verbosity)

    LOGGER.info("Config: %s", cfg)
    _require_environment()
    _log_library_versions()

    _set_seed(cfg.seed)
    _maybe_warn_about_grpo_batching(cfg)

    # ---------------------------------------------------------------
    # Load model/tokenizer (Unsloth) + prepare dataset
    # ---------------------------------------------------------------
    model, tokenizer = _load_unsloth_model(cfg)
    dataset = _prepare_dataset(cfg, tokenizer)

    max_prompt_length, max_completion_length = resolve_prompt_and_completion_lengths(cfg, dataset, tokenizer)
    LOGGER.info("Resolved max_prompt_length=%d, max_completion_length=%d", max_prompt_length, max_completion_length)

    # ---------------------------------------------------------------
    # Build GRPO trainer
    # ---------------------------------------------------------------
    from trl import GRPOConfig, GRPOTrainer

    use_bf16 = bool(torch.cuda.is_bf16_supported())

    # Build GRPOConfig kwargs in a version-tolerant way.
    grpo_kwargs: Dict[str, Any] = {
        # Core trainer args
        "output_dir": cfg.output_dir,
        "report_to": cfg.report_to,
        "seed": cfg.seed,
        "optim": cfg.optim,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_steps": cfg.max_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "remove_unused_columns": False,  # keep answer column for reward functions

        # Mixed precision
        "bf16": use_bf16,
        "fp16": (not use_bf16),

        # GRPO / sampling
        "num_generations": cfg.num_generations,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "repetition_penalty": cfg.repetition_penalty,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,

        # GRPO objective
        "beta": cfg.beta,
        "num_iterations": cfg.num_iterations,
        "epsilon": cfg.epsilon,
        "loss_type": cfg.loss_type,
        "scale_rewards": cfg.scale_rewards,
        "mask_truncated_completions": cfg.mask_truncated_completions,

        # Optional logging of sampled completions
        "log_completions": cfg.log_completions,
        "num_completions_to_print": cfg.num_completions_to_print,
    }

    grpo_kwargs = _filter_kwargs_for_callable(GRPOConfig, grpo_kwargs)
    training_args = GRPOConfig(**grpo_kwargs)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "reward_funcs": [
            reward_nonempty,
            reward_shaped_reference_match,
            reward_special_token_penalty,
        ],
        "args": training_args,
        "train_dataset": dataset,
    }

    # TRL is migrating from `tokenizer=` to `processing_class=`.
    try:
        trainer = GRPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = GRPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    # ---------------------------------------------------------------
    # Memory stats
    # ---------------------------------------------------------------
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved (before training).")

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------
    train_result = trainer.train()

    # ---------------------------------------------------------------
    # Final stats
    # ---------------------------------------------------------------
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_training = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    train_percentage = round(used_memory_for_training / max_memory * 100, 3)

    runtime_s = float(train_result.metrics.get("train_runtime", 0.0))
    print(f"{runtime_s} seconds used for training.")
    print(f"{round(runtime_s/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training delta = {used_memory_for_training} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory delta % of max memory = {train_percentage} %.")

    # ---------------------------------------------------------------
    # Quick generation demo
    # ---------------------------------------------------------------
    try:
        from transformers import TextStreamer

        if len(dataset) > 0:
            messages = dataset[0]["prompt"]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_text = prompt_text.removeprefix("<bos>")

            _ = model.generate(
                **tokenizer(prompt_text, return_tensors="pt").to("cuda"),
                max_new_tokens=min(256, max_completion_length),
                temperature=max(0.1, cfg.temperature),
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                streamer=TextStreamer(tokenizer, skip_prompt=True),
            )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Generation demo skipped: %s", exc)

    # ---------------------------------------------------------------
    # Save (PEFT adapter + tokenizer)
    # ---------------------------------------------------------------
    os.makedirs(cfg.save_dir, exist_ok=True)
    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    LOGGER.info("Saved to: %s", cfg.save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())