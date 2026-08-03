"""Provider-neutral research chunking components.

The callables accepted here may wrap local models or explicitly configured hosted
providers. ChunkKit never selects or calls a provider on its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from hashlib import sha256
from itertools import pairwise
from typing import Any

from .chunkers import _window_spans
from .errors import TokenBudgetError
from .models import Chunk, ChunkEdge, ChunkGraph, Document, TokenBudget
from .pipeline import ChunkingPipeline
from .protocols import Tokenizer

BoundaryScorer = Callable[[str, str], float]
PropositionExtractor = Callable[[Document], Sequence[tuple[int, int]]]
ContextProvider = Callable[[Chunk], str]


class SemanticChunker:
    """Split where adjacent sentence similarity falls below a threshold."""

    name = "semantic"
    _sentence = re.compile(r".+?(?:[.!?](?:[\"')\]]*)|$)(?:\s+|$)", re.DOTALL)

    def __init__(self, scorer: BoundaryScorer, *, threshold: float = 0.45) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        sentences = tuple(self._sentence.finditer(document.text))
        boundaries = {0, len(document.text)}
        scores: dict[int, float] = {}
        for left, right in pairwise(sentences):
            score = float(self.scorer(left.group().strip(), right.group().strip()))
            scores[left.end()] = score
            if score < self.threshold:
                boundaries.add(left.end())
        for start, end, metadata in _window_spans(document.text, tokenizer, budget, boundaries):
            nearby = [value for position, value in scores.items() if start < position <= end]
            yield (
                start,
                end,
                {
                    **metadata,
                    "semantic_threshold": self.threshold,
                    "minimum_adjacent_similarity": min(nearby) if nearby else None,
                },
            )


class PropositionChunker:
    """Chunk source-aligned proposition spans returned by an explicit extractor."""

    name = "proposition"

    def __init__(self, extractor: PropositionExtractor) -> None:
        self.extractor = extractor

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        for proposition, (start, end) in enumerate(self.extractor(document)):
            if not 0 <= start < end <= len(document.text):
                raise ValueError("proposition extractor returned an invalid source span")
            segment = document.text[start:end]
            for local_start, local_end, metadata in _window_spans(segment, tokenizer, budget):
                yield (
                    start + local_start,
                    start + local_end,
                    {
                        **metadata,
                        "proposition": proposition,
                    },
                )


class MultiResolutionPipeline:
    """Create coarse parents and fine children from the same source document."""

    def __init__(self, parent: ChunkingPipeline, child: ChunkingPipeline) -> None:
        if parent.budget.max_tokens <= child.budget.max_tokens:
            raise ValueError("the parent budget must be larger than the child budget")
        self.parent = parent
        self.child = child

    def graph(self, document: Document) -> ChunkGraph:
        parents = tuple(self.parent.chunk(document))
        children: list[Chunk] = []
        edges: list[ChunkEdge] = []
        for child in self.child.chunk(document):
            child_span = child.spans[0]
            containing = next(
                (
                    parent
                    for parent in parents
                    if parent.spans[0].start is not None
                    and parent.spans[0].end is not None
                    and child_span.start is not None
                    and child_span.end is not None
                    and parent.spans[0].start <= child_span.start
                    and parent.spans[0].end >= child_span.end
                ),
                None,
            )
            updated = child.model_copy(update={"parent_id": containing.id if containing else None})
            children.append(updated)
            if containing:
                edges.extend(
                    (
                        ChunkEdge(source=updated.id, target=containing.id, relation="parent"),
                        ChunkEdge(source=containing.id, target=updated.id, relation="child"),
                    )
                )
        return ChunkGraph(chunks=parents + tuple(children), edges=tuple(edges))


class ContextualEnricher:
    """Prefix chunks with explicit contextual text while enforcing a target budget."""

    def __init__(
        self, provider: ContextProvider, tokenizer: Tokenizer, budget: TokenBudget
    ) -> None:
        self.provider = provider
        self.tokenizer = tokenizer
        self.budget = budget

    def enrich(self, chunks: Iterable[Chunk]) -> Iterable[Chunk]:
        for chunk in chunks:
            context = self.provider(chunk).strip()
            text = f"{context}\n\n{chunk.text}" if context else chunk.text
            count = self.tokenizer.count(text)
            if count > self.budget.max_tokens:
                raise TokenBudgetError(
                    f"Contextual enrichment would emit {count} tokens for a "
                    f"{self.budget.max_tokens}-token budget"
                )
            identifier = sha256(f"{chunk.id}\0{context}".encode()).hexdigest()
            yield chunk.model_copy(
                update={
                    "id": identifier,
                    "text": text,
                    "token_counts": {**chunk.token_counts, self.tokenizer.name: count},
                    "metadata": {
                        **chunk.metadata,
                        "contextual_enrichment": bool(context),
                        "original_chunk_id": chunk.id,
                    },
                }
            )
