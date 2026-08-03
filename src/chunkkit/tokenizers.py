"""Dependency-free tokenizers plus optional exact provider tokenizers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .errors import ConfigurationError, missing_extra


class UnicodeTokenizer:
    """A deterministic, dependency-free word/punctuation tokenizer.

    It is intended for offline pipelines and tests. Provider models should use an
    exact provider tokenizer or token-counting endpoint.
    """

    name = "unicode"
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def spans(self, text: str) -> Sequence[tuple[int, int]]:
        return tuple((match.start(), match.end()) for match in self._pattern.finditer(text))

    def count(self, text: str) -> int:
        return len(self.spans(text))


class CharacterTokenizer:
    name = "character"

    def spans(self, text: str) -> Sequence[tuple[int, int]]:
        return tuple((index, index + 1) for index in range(len(text)))

    def count(self, text: str) -> int:
        return len(text)


class TiktokenTokenizer:
    def __init__(self, encoding: str = "cl100k_base", *, model: str | None = None) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise missing_extra("TiktokenTokenizer", "openai", "tiktoken") from exc
        self._encoding = (
            tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding(encoding)
        )
        self.name = f"tiktoken:{self._encoding.name}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def spans(self, text: str) -> Sequence[tuple[int, int]]:
        if not text:
            return ()
        tokens = self._encoding.encode(text, disallowed_special=())
        spans: list[tuple[int, int]] = []
        byte_cursor = 0
        encoded = text.encode("utf-8")
        for token in tokens:
            token_bytes = self._encoding.decode_single_token_bytes(token)
            start_byte = byte_cursor
            byte_cursor += len(token_bytes)
            start = len(encoded[:start_byte].decode("utf-8", errors="ignore"))
            end = len(encoded[:byte_cursor].decode("utf-8", errors="ignore"))
            spans.append((start, max(start + 1, end)))
        return tuple(spans)


class HuggingFaceTokenizer:
    def __init__(self, model: str, *, trust_remote_code: bool = False) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise missing_extra("HuggingFaceTokenizer", "huggingface", "transformers") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=True, trust_remote_code=trust_remote_code
        )
        self.name = f"huggingface:{model}"

    def spans(self, text: str) -> Sequence[tuple[int, int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ConfigurationError(
                "The selected Hugging Face tokenizer does not expose offset mappings"
            )
        return tuple((int(start), int(end)) for start, end in offsets if end > start)

    def count(self, text: str) -> int:
        return len(self.spans(text))


def create_tokenizer(name: str, *, model: str | None = None):  # type: ignore[no-untyped-def]
    if name == "unicode":
        return UnicodeTokenizer()
    if name == "character":
        return CharacterTokenizer()
    if name.startswith("tiktoken:"):
        return TiktokenTokenizer(name.partition(":")[2], model=model)
    if name.startswith("huggingface:"):
        return HuggingFaceTokenizer(name.partition(":")[2])
    if name in {"cl100k_base", "o200k_base", "p50k_base", "r50k_base"}:
        return TiktokenTokenizer(name, model=model)
    raise ConfigurationError(
        f"Unknown tokenizer '{name}'. Register a 'chunkkit.tokenizers' plugin or provide "
        "one of: unicode, character, tiktoken:<encoding>."
    )
