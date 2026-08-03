"""Deterministic built-in chunking strategies."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .models import Document, TokenBudget
from .protocols import Chunker, Tokenizer


def _preferred_boundaries(text: str, patterns: Sequence[re.Pattern[str]]) -> set[int]:
    boundaries = {0, len(text)}
    for pattern in patterns:
        boundaries.update(match.end() for match in pattern.finditer(text))
    return boundaries


def _window_spans(
    text: str,
    tokenizer: Tokenizer,
    budget: TokenBudget,
    boundaries: set[int] | None = None,
) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
    token_spans = list(tokenizer.spans(text))
    if not token_spans:
        return
    token_index = 0
    ordinal = 0
    while token_index < len(token_spans):
        limit_index = min(token_index + budget.max_tokens, len(token_spans))
        start = token_spans[token_index][0]
        end = token_spans[limit_index - 1][1]
        if boundaries and limit_index < len(token_spans):
            candidates = [point for point in boundaries if start < point <= end]
            if candidates:
                preferred_end = max(candidates)
                preferred_count = sum(
                    1 for left, _ in token_spans[token_index:] if left < preferred_end
                )
                # Avoid tiny chunks when a weak separator appears near the start of the
                # window. Structural boundaries are preferred only when they use at least
                # three quarters of the available budget.
                if preferred_count > 0 and preferred_count * 4 >= budget.max_tokens * 3:
                    limit_index = min(token_index + preferred_count, len(token_spans))
                    end = token_spans[limit_index - 1][1]

        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            yield start, end, {"continuation": ordinal > 0}
            ordinal += 1
        if limit_index >= len(token_spans):
            break
        token_index = max(token_index + 1, limit_index - budget.overlap_tokens)


class FixedTokenChunker:
    name = "token"

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        return _window_spans(document.text, tokenizer, budget)


class CharacterChunker(FixedTokenChunker):
    name = "character"


class WordChunker(FixedTokenChunker):
    name = "word"


class SlidingWindowChunker(FixedTokenChunker):
    name = "sliding"


class RecursiveChunker:
    name = "recursive"
    _patterns = (
        re.compile(r"\n\s*\n"),
        re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+"),
        re.compile(r"[,;:]\s+"),
        re.compile(r"\s+"),
    )

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        boundaries = _preferred_boundaries(document.text, self._patterns)
        return _window_spans(document.text, tokenizer, budget, boundaries)


class SentenceChunker:
    name = "sentence"
    _patterns = (re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+"),)

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        return _window_spans(
            document.text, tokenizer, budget, _preferred_boundaries(document.text, self._patterns)
        )


class ParagraphChunker:
    name = "paragraph"
    _patterns = (re.compile(r"\n\s*\n"),)

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        return _window_spans(
            document.text, tokenizer, budget, _preferred_boundaries(document.text, self._patterns)
        )


class MarkdownChunker:
    name = "markdown"
    _patterns = (re.compile(r"(?m)(?=^#{1,6}\s)"), re.compile(r"\n\s*\n"))

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        return _window_spans(
            document.text, tokenizer, budget, _preferred_boundaries(document.text, self._patterns)
        )


class CodeChunker:
    """Dependency-free structural fallback; Tree-sitter plugins can replace it."""

    name = "code"
    _patterns = (
        re.compile(r"(?m)(?=^(?:async\s+)?(?:def|class|function|interface|struct|enum)\s+)"),
        re.compile(r"\n\s*\n"),
    )

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        return _window_spans(
            document.text, tokenizer, budget, _preferred_boundaries(document.text, self._patterns)
        )


class AdaptiveChunker:
    name = "adaptive"

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        mime = document.mime_type.lower()
        strategy: Chunker
        if "markdown" in mime or document.source_uri.endswith((".md", ".mdx")):
            strategy = MarkdownChunker()
        elif mime.startswith("text/x-") or document.source_uri.endswith(
            (".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp")
        ):
            strategy = CodeChunker()
        else:
            strategy = RecursiveChunker()
        for start, end, metadata in strategy.split(document, budget, tokenizer):
            yield start, end, {**metadata, "selected_strategy": strategy.name}


BUILTIN_CHUNKERS = {
    item.name: item
    for item in (
        FixedTokenChunker(),
        CharacterChunker(),
        WordChunker(),
        SlidingWindowChunker(),
        RecursiveChunker(),
        SentenceChunker(),
        ParagraphChunker(),
        MarkdownChunker(),
        CodeChunker(),
        AdaptiveChunker(),
    )
}
