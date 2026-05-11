"""Local LLM wrapper (HuggingFace transformers)."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenmind.config import LLM_MODEL_NAME, MAX_CONTEXT_LENGTH, hf_token


class LocalLLM:
    """Tokenizer + causal model pair with chat-templated answer generation.

    The HF token is read from ``HF_TOKEN`` via :mod:`eigenmind.config` — never hard-coded.
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME, *, lazy: bool = True):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        if not lazy:
            self._load()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _load(self) -> None:
        token = hf_token()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                token=token,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Error loading LLM model: {e}") from e

    def answer(
        self,
        prompt: str,
        context: str,
        source_details: list[dict] | None = None,
    ) -> tuple[str, list[str]]:
        """Generate an answer grounded in ``context`` and quote the sources used.

        Returns ``(assistant_response, formatted_references)``.
        """
        if not context.strip():
            return "Context is empty. Cannot generate an answer.", []

        tokenizer = self.tokenizer
        model = self.model

        max_new_tokens = 512
        model_max_length = min(getattr(tokenizer, "model_max_length", 4096), MAX_CONTEXT_LENGTH)

        overhead = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        overhead_tokens = len(tokenizer.encode(overhead))
        available = model_max_length - overhead_tokens - max_new_tokens - 50

        ctx_tokens = tokenizer.encode(context)
        if len(ctx_tokens) > available:
            context = tokenizer.decode(ctx_tokens[:available], skip_special_tokens=True)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant. Consider the user's prompt (question, "
                    "assertion or topic of interest) and explain to what extent the provided "
                    "context is related to the prompt. Do not use any external knowledge. "
                    "You must quote parts of the context to support your answer."
                    f"\n\nContext:\n{context}"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        prompt_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)

        input_len = inputs.input_ids.shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

        formatted_refs = []
        for i, details in enumerate(source_details or []):
            tags = details.get("tags") or (["Singular"] if details.get("is_singular") else [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            snippet = details.get("text", "N/A").replace("?", "'")
            formatted_refs.append(
                f"[{i + 1}] Source: {details['filename']}{tag_str}\n"
                f'    Context: "{snippet}"'
            )
        return response, formatted_refs
