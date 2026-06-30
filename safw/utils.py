"""Batch generation and model loading for SAF-W and the baselines.

This module provides the shared inference utilities used by the entry point:

- :class:`KeyWordsCriteria`, a stopping criterion that halts a sequence once a
  stop string appears;
- :func:`generate_completions`, batched generation over a list of prompts with
  stopping criteria and optional logits processors;
- :func:`add_pad_token`, which ensures a tokenizer has a pad token;
- :func:`load_trimix`, :func:`load_proxy`, and :func:`load_safw`, which build
  the three decoder types.

Models load in bfloat16 by default in this thesis. The original TriMix release
loads in 8-bit; pass ``load_in_8bit=True`` to recover that behaviour.
"""

from typing import List
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .dexperts import DExpertsLlama, TriMixQwen, SAFW


class KeyWordsCriteria(StoppingCriteria):
    """Stop generation for a sequence once any stop id-sequence appears.

    :param stop_id_sequences: A list of token-id lists; if the tail of a
        generated sequence matches any of them, that sequence is finished.
    """

    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), \
            "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        """Return True only when every sequence in the batch has stopped.

        :param input_ids: The running token ids, shape ``(batch, L)``.
        :param scores: Unused.
        :returns: True if all sequences have produced a stop sequence.
        """
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    stopped = True
                    break
            sequences_should_be_stopped.append(stopped)
        return all(sequences_should_be_stopped)


def add_pad_token(tokenizer, padding_side="left"):
    """Ensure the tokenizer has a pad token and a padding side.

    :param tokenizer: The tokenizer to adjust.
    :param padding_side: Padding side to set.
    :returns: The adjusted tokenizer.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 1,
    stop_id_sequences=None,
    add_special_tokens: bool = True,
    disable_tqdm: bool = False,
    **generation_kwargs,
):
    """Generate completions for a list of prompts in batches.

    Prompts are tokenized with left padding, generated in batches with a
    :class:`KeyWordsCriteria` stopping criterion, and decoded. Only the newly
    generated tokens are returned (the prompt is stripped). When a stop
    sequence is given, the completion is truncated at its first occurrence.

    :param model: A decoder exposing a ``generate(input_ids, ...)`` method.
    :param tokenizer: The shared tokenizer.
    :param prompts: The list of prompt strings.
    :param batch_size: Number of prompts per batch.
    :param stop_id_sequences: List of stop token-id sequences, or ``None``.
    :param add_special_tokens: Whether to add special tokens when encoding.
    :param disable_tqdm: Disable the progress bar.
    :param generation_kwargs: Forwarded to ``model.generate``.
    :returns: A list of generated strings, one per prompt.
    """
    generations = []
    progress = tqdm(total=len(prompts), disable=disable_tqdm)

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        tokenized = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)

        stopping_criteria = (
            StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)])
            if stop_id_sequences else None
        )

        try:
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stopping_criteria=stopping_criteria,
                **generation_kwargs,
            )

            # Strip the prompt; keep only generated tokens.
            gen_only = output_ids[:, input_ids.shape[1]:]

            # Truncate at the first stop sequence, per example.
            if stop_id_sequences:
                for j in range(gen_only.shape[0]):
                    ids = gen_only[j].tolist()
                    cut = len(ids)
                    for k in range(len(ids)):
                        for stop in stop_id_sequences:
                            if ids[k : k + len(stop)] == stop:
                                cut = k
                                break
                        if cut != len(ids):
                            break
                    gen_only[j, cut:] = tokenizer.pad_token_id

            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            print(f"batch {i} failed: {exc}")
            decoded = [""] * len(batch_prompts)

        generations.extend(decoded)
        progress.update(len(batch_prompts))

    progress.close()
    assert len(generations) == len(prompts)
    return generations


def _model_kwargs(load_in_8bit: bool) -> dict:
    """Build model kwargs: bfloat16 by default, 8-bit if requested."""
    if load_in_8bit:
        return {"load_in_8bit": True, "device_map": "auto"}
    return {"torch_dtype": torch.bfloat16, "device_map": "auto"}


def load_trimix(
    base_model_name_or_path,
    expert_model_name_or_path,
    antiexpert_model_name_or_path,
    base_weight,
    expert_weight,
    plausibility_model="expert",
    plausibility_alpha=0.1,
    load_in_8bit=False,
    padding_side="left",
):
    """Load a TriMix decoder and its tokenizer.

    :returns: A tuple ``(model, tokenizer)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer = add_pad_token(tokenizer, padding_side)
    model = TriMixQwen(
        base_model_name_or_path=base_model_name_or_path,
        expert_model_name_or_path=expert_model_name_or_path,
        antiexpert_model_name_or_path=antiexpert_model_name_or_path,
        tokenizer=tokenizer,
        base_weight=base_weight,
        expert_weight=expert_weight,
        plausibility_model=plausibility_model,
        plausibility_alpha=plausibility_alpha,
        model_kwargs=_model_kwargs(load_in_8bit),
    )
    return model, tokenizer


def load_proxy(
    base_model_name_or_path,
    expert_model_name_or_path,
    antiexpert_model_name_or_path,
    alpha=1.0,
    load_in_8bit=False,
    padding_side="left",
):
    """Load a Proxy Tuning decoder and its tokenizer.

    :returns: A tuple ``(model, tokenizer)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer = add_pad_token(tokenizer, padding_side)
    model = DExpertsLlama(
        base_model_name_or_path=base_model_name_or_path,
        expert_model_name_or_path=expert_model_name_or_path,
        antiexpert_model_name_or_path=antiexpert_model_name_or_path,
        tokenizer=tokenizer,
        alpha=alpha,
        model_kwargs=_model_kwargs(load_in_8bit),
    )
    return model, tokenizer


def load_safw(
    host_model_name_or_path,
    scorer_model_name_or_path,
    beta_fixed=0.5,
    fixed_beta=False,
    load_in_8bit=False,
    padding_side="left",
):
    """Load a SAF-W decoder and its tokenizer.

    :param host_model_name_or_path: Host model path or hub id.
    :param scorer_model_name_or_path: Scorer model path or hub id.
    :param beta_fixed: Constant scorer weight for uniform averaging (used when
        ``fixed_beta`` is True).
    :param fixed_beta: Use a constant beta (uniform averaging) instead of the
        endorsement weight.
    :param load_in_8bit: Load in 8-bit instead of bfloat16.
    :param padding_side: Tokenizer padding side.
    :returns: A tuple ``(model, tokenizer)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(host_model_name_or_path)
    tokenizer = add_pad_token(tokenizer, padding_side)
    model = SAFW(
        host_model_name_or_path=host_model_name_or_path,
        scorer_model_name_or_path=scorer_model_name_or_path,
        tokenizer=tokenizer,
        beta_fixed=beta_fixed,
        fixed_beta=fixed_beta,
        model_kwargs=_model_kwargs(load_in_8bit),
    )
    return model, tokenizer
