from __future__ import annotations

from chunkkit import (
    ChunkingPipeline,
    ContextPacker,
    Document,
    ModelTarget,
    PipelineSpec,
    ScoredChunk,
)


def make_chunks():  # type: ignore[no-untyped-def]
    pipeline = ChunkingPipeline.from_spec(
        PipelineSpec(
            chunker="token",
            chunk_size=3,
            overlap=0,
            target=ModelTarget(max_input_tokens=20, reserved_tokens=1, safety_margin_tokens=1),
        )
    )
    return tuple(
        pipeline.chunk(Document(text="one two three four five six seven", source_uri="x://a"))
    )


def test_context_packer_ranks_deduplicates_and_cites() -> None:
    chunks = make_chunks()
    target = ModelTarget(max_input_tokens=8, reserved_tokens=1, safety_margin_tokens=1)
    bundle = ContextPacker(target).assemble(
        [ScoredChunk(chunk=chunks[0], score=1), ScoredChunk(chunk=chunks[0], score=0), chunks[1]]
    )
    assert bundle.chunks[0].id == chunks[0].id
    assert any(item.reason == "duplicate" for item in bundle.dropped)
    assert bundle.token_count <= bundle.max_tokens
    assert bundle.citations[0].source_uri == "x://a"


def test_context_packer_reports_budget_drops() -> None:
    chunks = make_chunks()
    target = ModelTarget(max_input_tokens=5, reserved_tokens=1, safety_margin_tokens=1)
    bundle = ContextPacker(target).assemble(list(chunks))
    assert any(item.reason == "budget" for item in bundle.dropped)
