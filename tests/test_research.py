from __future__ import annotations

import pytest

from chunkkit import ChunkingPipeline, Document, ModelTarget, PipelineSpec, TokenBudgetError
from chunkkit.research import (
    ContextualEnricher,
    MultiResolutionPipeline,
    PropositionChunker,
    SemanticChunker,
)
from chunkkit.tokenizers import UnicodeTokenizer


def make_pipeline(size: int) -> ChunkingPipeline:
    return ChunkingPipeline.from_spec(
        PipelineSpec(
            chunk_size=size,
            overlap=0,
            target=ModelTarget(
                max_input_tokens=size + 8,
                reserved_tokens=1,
                safety_margin_tokens=1,
            ),
        )
    )


def test_semantic_chunker_uses_explicit_scorer() -> None:
    def scorer(left: str, right: str) -> float:
        return 0.1 if "Cats" in left and "Rockets" in right else 0.9

    chunker = SemanticChunker(scorer, threshold=0.5)
    pipeline = ChunkingPipeline(
        make_pipeline(8).spec,
        chunker,
        UnicodeTokenizer(),
    )
    chunks = tuple(pipeline.chunk(Document(text="Cats purr. Rockets launch. Space is vast.")))
    assert chunks
    assert all(chunk.metadata["semantic_threshold"] == 0.5 for chunk in chunks)


def test_proposition_chunker_keeps_source_spans() -> None:
    document = Document(text="First fact. Second fact.")

    def extractor(value: Document) -> tuple[tuple[int, int], ...]:
        return ((0, 11), (12, len(value.text)))

    pipeline = ChunkingPipeline(
        make_pipeline(4).spec,
        PropositionChunker(extractor),
        UnicodeTokenizer(),
    )
    chunks = tuple(pipeline.chunk(document))
    assert [chunk.metadata["proposition"] for chunk in chunks] == [0, 1]
    assert all(document.text[c.spans[0].start : c.spans[0].end] == c.text for c in chunks)


def test_multi_resolution_graph_links_children() -> None:
    graph = MultiResolutionPipeline(make_pipeline(8), make_pipeline(3)).graph(
        Document(text="one two three four five six seven eight")
    )
    assert any(edge.relation == "parent" for edge in graph.edges)
    assert any(chunk.parent_id for chunk in graph.chunks)


def test_contextual_enrichment_is_budgeted() -> None:
    chunks = tuple(make_pipeline(4).chunk(Document(text="one two")))
    tokenizer = UnicodeTokenizer()
    enriched = tuple(
        ContextualEnricher(
            lambda chunk: "document context",
            tokenizer,
            make_pipeline(5).budget,
        ).enrich(chunks)
    )
    assert enriched[0].text.startswith("document context")
    assert enriched[0].metadata["original_chunk_id"] == chunks[0].id
    with pytest.raises(TokenBudgetError):
        tuple(
            ContextualEnricher(
                lambda chunk: "too much context here",
                tokenizer,
                make_pipeline(2).budget,
            ).enrich(chunks)
        )
