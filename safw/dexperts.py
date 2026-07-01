"""Logit-fusion decoders: Proxy Tuning, TriMix, and SAF-W.

This module is built on the TriMix / DExperts decoding framework. The base
class :class:`DExpertsLlama` implements Proxy Tuning. :class:`TriMixQwen`
implements TriMix. :class:`SAFW` implements the method of this thesis,
Scorer-Adaptive Fusion with Weighting, by reusing the same KV-cached
generation loop and changing only the fusion rule.

All methods share the same generation machinery: cached decoding via
``past_key_values`` and ``prepare_inputs_for_generation``, batch generation
with per-sequence finishing, stopping criteria, logits processors, and
vocabulary alignment between models of different vocab size. Only the way the
next-token logits are combined differs between methods.

Models load in bfloat16 by default in this thesis; the original TriMix release
loads in 8-bit (pass ``load_in_8bit`` through ``model_kwargs``).
"""

from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList,
)
from torch import Tensor
from collections import defaultdict

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus filtering.

    :param logits: Logits of shape ``(batch, vocab)``.
    :param top_k: Keep only the top-k tokens if ``> 0``.
    :param top_p: Keep the smallest set with cumulative probability ``>= top_p``.
    :param filter_value: Value assigned to filtered positions.
    :param min_tokens_to_keep: Minimum tokens kept per example.
    :returns: The filtered logits.
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


class DExpertsLlama:
    """Proxy Tuning decoder over a base, expert, and antiexpert model.

    The name is inherited from the DExperts / Proxy-Tuning framework this code
    builds on; the class is model-agnostic and works with Qwen, Llama, and
    Gemma. The Qwen experiments in this thesis use the :class:`TriMixQwen` and
    :class:`SAFW` subclasses.

    The fused next-token logits are
    ``L_base + alpha * (L_expert - L_antiexpert)``. Decoding is KV-cached and
    supports batches and stopping criteria.

    :param base_model_name_or_path: Path or hub id of the base (large) model.
    :param expert_model_name_or_path: Path or hub id of the expert (small-cpt).
    :param antiexpert_model_name_or_path: Path or hub id of the antiexpert
        (small-base).
    :param tokenizer: Shared tokenizer.
    :param system_prompt: Optional system prompt for chat-format experts.
    :param alpha: Residual weight.
    :param chat_response_prefix: Optional response prefix for chat experts.
    :param model_kwargs: Keyword arguments forwarded to ``from_pretrained``.
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        alpha: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
    ):
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path, **model_kwargs
        )
        self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path, **model_kwargs
        )

        self.base.eval()
        self.expert.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        self.use_chat_format_for_expert = (
            True if "chat" in expert_model_name_or_path.lower() else False
        )

        if self.use_chat_format_for_expert:
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"
            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"
            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

    def forward(self, base_inputs, expert_inputs, antiexpert_inputs,
                return_dict=None):
        """Run one forward pass of all three models.

        :param base_inputs: Prepared inputs for the base model.
        :param expert_inputs: Prepared inputs for the expert model.
        :param antiexpert_inputs: Prepared inputs for the antiexpert model.
        :param return_dict: Whether to return a dict output.
        :returns: A tuple ``(base_outputs, expert_outputs, antiexpert_outputs)``.
        """
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)
        return base_outputs, expert_outputs, antiexpert_outputs

    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode and re-encode inputs with chat formatting for the expert.

        :param input_ids: The current input ids.
        :returns: Tokenized chat inputs on the model device.
        """
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        if self.chat_response_prefix:
            cleaned = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, "").rstrip()
                cleaned.append(p)
        else:
            cleaned = prompts
        chat_prompts = [f"{self.chat_prefix} {p} {self.chat_suffix}" for p in cleaned]
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)
        return chat_inputs

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        """Record per-step tokens and logits when analysis is requested.

        :param analysis_data: The accumulating analysis dict.
        :param next_tokens: The chosen tokens this step.
        :param next_token_logits_dict: Per-model logits this step.
        :returns: The updated analysis dict.
        """
        analysis_data["tokens"].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data["token_ids"].append(next_tokens)
        for model in next_token_logits_dict.keys():
            analysis_data[f"logits_{model}"].append(
                next_token_logits_dict[model].unsqueeze(dim=1)
            )
        return analysis_data

    def fuse_logits(self, base_logits, expert_logits, antiexpert_logits):
        """Combine the three models' next-token logits (Proxy Tuning rule).

        Subclasses override this to implement a different fusion. The base
        implementation is ``L_base + alpha * (L_expert - L_antiexpert)``.

        :param base_logits: Base next-token logits, shape ``(batch, vocab)``.
        :param expert_logits: Expert next-token logits (already truncated to
            the base vocab), shape ``(batch, vocab)``.
        :param antiexpert_logits: Antiexpert next-token logits, shape
            ``(batch, vocab)``.
        :returns: The fused logits, shape ``(batch, vocab)``.
        """
        return base_logits + self.alpha * (expert_logits - antiexpert_logits)

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs,
    ):
        """Greedily (or with sampling) decode with cached three-model fusion.

        :param input_ids: Prompt token ids, shape ``(batch, L)``.
        :param max_new_tokens: Maximum tokens to generate.
        :param do_sample: Sample instead of greedy argmax.
        :param top_p: Nucleus sampling threshold.
        :param temperature: Sampling temperature.
        :param logits_processor: Optional logits processors.
        :param stopping_criteria: Optional stopping criteria.
        :param return_logits_for_analysis: Also return per-step logits.
        :returns: The output ids, or ``(ids, analysis_data)`` if requested.
        """
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            expert_kwargs["attention_mask"] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(
            input_ids.device
        )

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        for step in range(max_new_tokens):
            self.base._supports_cache_class = False
            self.expert._supports_cache_class = False
            self.antiexpert._supports_cache_class = False

            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(
                expert_input_ids, **expert_kwargs
            )
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(
                input_ids, **antiexpert_kwargs
            )

            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]

            expert_next_token_logits = expert_next_token_logits[
                :, : base_next_token_logits.shape[-1]
            ]
            antiexpert_next_token_logits = antiexpert_next_token_logits[
                :, : base_next_token_logits.shape[-1]
            ]

            next_token_logits = self.fuse_logits(
                base_next_token_logits,
                expert_next_token_logits,
                antiexpert_next_token_logits,
            )

            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = (
                next_tokens * unfinished_sequences
                + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    "dexperts": next_token_logits,
                    "base": base_next_token_logits,
                    "expert": expert_next_token_logits,
                    "antiexpert": antiexpert_next_token_logits,
                }
                analysis_data = self.update_analysis_data(
                    analysis_data, next_tokens, next_token_logits_dict
                )

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = torch.cat(
                [expert_input_ids, next_tokens[:, None]], dim=-1
            )

            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs = self._update_model_kwargs_for_generation(
                expert_outputs, expert_kwargs
            )
            antiexpert_kwargs = self._update_model_kwargs_for_generation(
                antiexpert_outputs, antiexpert_kwargs
            )

            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
            if unfinished_sequences.max() == 0:
                break

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith("logits"):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Carry ``past_key_values`` and extend the attention mask one step.

        :param outputs: The model outputs from this step.
        :param kwargs: The kwargs to update.
        :returns: The updated kwargs.
        """
        kwargs["past_key_values"] = outputs.past_key_values
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
        return kwargs


class TriMixQwen(DExpertsLlama):
    """TriMix decoder for Qwen and Llama models.

    The fused logits are
    ``w_base * L_base + w_expert * L_expert + (1 - w_base - w_expert) * L_antiexpert``
    with an optional plausibility constraint that masks tokens the plausibility
    model finds unlikely. Expert/antiexpert vocabularies are resized to the
    base vocabulary on construction.

    :param base_weight: Weight on the base logits.
    :param expert_weight: Weight on the expert logits.
    :param plausibility_model: ``"expert"``, ``"antiexpert"``, ``"base"`` or
        anything else to disable the constraint.
    :param plausibility_alpha: Plausibility threshold.
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        base_weight: float = 1.0,
        expert_weight: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        plausibility_model: str = "expert",
        plausibility_alpha: float = 0.1,
    ):
        super().__init__(
            base_model_name_or_path, expert_model_name_or_path,
            antiexpert_model_name_or_path, tokenizer, system_prompt,
            base_weight, chat_response_prefix, model_kwargs,
        )
        if self.base.get_input_embeddings().weight.size(0) != self.expert.get_input_embeddings().weight.size(0):
            self.expert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))
            self.antiexpert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))

        self.plausibility_model = plausibility_model
        self.plausibility_alpha = plausibility_alpha
        self.base_weight = base_weight
        self.expert_weight = expert_weight
        self._plausibility_alpha_tensor = None

    def fuse_logits(self, base_logits, expert_logits, antiexpert_logits):
        """Weighted three-model sum with optional plausibility masking."""
        next_token_logits = (
            self.base_weight * base_logits
            + self.expert_weight * expert_logits
            + (1 - self.base_weight - self.expert_weight) * antiexpert_logits
        )
        if self.plausibility_model in ("expert", "antiexpert", "base"):
            if self._plausibility_alpha_tensor is None:
                self._plausibility_alpha_tensor = torch.tensor(
                    self.plausibility_alpha, device=base_logits.device
                )
            source = {
                "expert": expert_logits,
                "antiexpert": antiexpert_logits,
                "base": base_logits,
            }[self.plausibility_model]
            max_logit = source.max(dim=-1).values
            mask = source > torch.log(self._plausibility_alpha_tensor) + max_logit.unsqueeze(dim=-1)
            next_token_logits[~mask] = -float("inf")
        return next_token_logits


# TriMix for Gemma uses the same fusion; the original keeps a separate class
# for Gemma's no-cache forward. TriMixQwen covers Qwen and Llama as used here.
TriMixGemma = TriMixQwen


class SAFW(DExpertsLlama):
    """Scorer-Adaptive Fusion with Weighting (the method of this thesis).

    SAF-W combines two models, a host and a scorer. The host proposes its top
    token; the scorer reports the probability it assigns to that token (the
    endorsement ``e``). The scorer weight is ``beta = 1 - e`` and the fused
    logits are ``(1 - beta) * L_host + beta * L_scorer``.

    SAF-W reuses the cached three-model generation loop by mapping the host to
    the ``base`` slot and the scorer to the ``expert`` slot; the antiexpert
    slot is loaded with the scorer as well so the loop runs unchanged, but only
    the host and scorer logits enter the fusion. Which model is host and which
    is scorer is selected per task and language on a development set; it is not
    fixed.

    :param host_model_name_or_path: Path or hub id of the host model.
    :param scorer_model_name_or_path: Path or hub id of the scorer model.
    :param tokenizer: Shared tokenizer.
    :param model_kwargs: Keyword arguments forwarded to ``from_pretrained``.
    :param beta_fixed: Constant scorer weight for the uniform-averaging
        reduction; only used when ``fixed_beta`` is True.
    :param fixed_beta: If True, use a constant ``beta = beta_fixed`` (0.5 is
        uniform averaging, the delta = 0 reduction used for math). If False,
        use the endorsement weight ``beta = 1 - e``.
    """

    def __init__(
        self,
        host_model_name_or_path: str,
        scorer_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        beta_fixed: float = 0.5,
        fixed_beta: bool = False,
    ):
        super().__init__(
            base_model_name_or_path=host_model_name_or_path,
            expert_model_name_or_path=scorer_model_name_or_path,
            antiexpert_model_name_or_path=scorer_model_name_or_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            alpha=1.0,
            chat_response_prefix=chat_response_prefix,
            model_kwargs=model_kwargs,
        )
        if self.base.get_input_embeddings().weight.size(0) != self.expert.get_input_embeddings().weight.size(0):
            self.expert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))
            self.antiexpert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))

        # SAF-W only uses host (base slot) and scorer (expert slot). The parent
        # loaded a second scorer copy into the antiexpert slot so the 3-model
        # loop runs unchanged, but fuse_logits ignores antiexpert. Release that
        # duplicate and point antiexpert at the same scorer object to save
        # memory (one 1.5B copy instead of two).
        import gc
        del self.antiexpert
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.antiexpert = self.expert

        self.beta_fixed = beta_fixed
        self.fixed_beta = fixed_beta

    def fuse_logits(self, base_logits, expert_logits, antiexpert_logits):
        """Endorsement fusion of host (base slot) and scorer (expert slot).

        The antiexpert slot is ignored. With ``fixed_beta`` the scorer weight
        is the constant ``beta_fixed`` (uniform averaging at 0.5); otherwise it
        is ``beta = 1 - e`` where ``e`` is the scorer probability of the host's
        top token.
        """
        l_host = base_logits
        l_scorer = expert_logits
        if self.fixed_beta:
            beta = self.beta_fixed
        else:
            p_scorer = F.softmax(l_scorer, dim=-1)
            host_top1 = l_host.argmax(dim=-1, keepdim=True)
            e = p_scorer.gather(-1, host_top1).squeeze(-1)
            beta = (1.0 - e).unsqueeze(-1)
        return (1.0 - beta) * l_host + beta * l_scorer
