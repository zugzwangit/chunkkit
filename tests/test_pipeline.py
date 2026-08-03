from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from chunkkit import (
    Acl,
    AclResolution,
    ChunkingPipeline,
    Document,
    IncompleteAclError,
    ModelTarget,
    PipelineSpec,
    Visibility,
)


def pipeline(size: int = 8, overlap: int = 2, chunker: str = "recursive") -> ChunkingPipeline:
    return ChunkingPipeline.from_spec(
        PipelineSpec(
            chunker=chunker,
            chunk_size=size,
            overlap=overlap,
            target=ModelTarget(
                tokenizer="unicode",
                max_input_tokens=max(size + 2, 32),
                reserved_tokens=1,
                safety_margin_tokens=1,
            ),
        )
    )


def test_public_quickstart_and_graph() -> None:
    document = Document(
        text="One two three four five six. Seven eight nine ten.", source_uri="x://1"
    )
    chunks = tuple(pipeline().chunk(document))
    assert chunks
    assert all(chunk.token_counts["unicode"] <= 8 for chunk in chunks)
    assert chunks[0].previous_id is None
    assert chunks[-1].next_id is None
    assert pipeline().graph(document).edges


def test_ids_are_deterministic_and_recipe_sensitive() -> None:
    document = Document(
        text="alpha beta gamma delta epsilon", source_uri="memory://same", revision="1"
    )
    first = [chunk.id for chunk in pipeline(3, 1).chunk(document)]
    second = [chunk.id for chunk in pipeline(3, 1).chunk(document)]
    changed = [chunk.id for chunk in pipeline(4, 1).chunk(document)]
    assert first == second
    assert first != changed


@given(st.text(min_size=1, max_size=500))
def test_every_chunk_obeys_budget(text: str) -> None:
    chunks = tuple(pipeline(12, 3).chunk(Document(text=text)))
    assert all(chunk.token_counts["unicode"] <= 12 for chunk in chunks)
    for chunk in chunks:
        span = chunk.spans[0]
        assert span.start is not None and span.end is not None
        assert text[span.start : span.end] == chunk.text


@pytest.mark.parametrize(
    "strategy", ["token", "sliding", "sentence", "paragraph", "markdown", "code", "adaptive"]
)
def test_builtin_strategies(strategy: str) -> None:
    document = Document(
        text="# Heading\n\ndef function():\n    return 1\n\nA sentence. Another sentence.",
        mime_type="text/markdown",
    )
    assert tuple(pipeline(10, 1, strategy).chunk(document))


def test_incomplete_restricted_acl_fails_closed() -> None:
    document = Document(
        text="secret",
        acl=Acl(visibility=Visibility.RESTRICTED, resolution=AclResolution.INCOMPLETE),
    )
    with pytest.raises(IncompleteAclError):
        tuple(pipeline().chunk(document))

    spec = pipeline().spec.model_copy(update={"allow_incomplete_acl": True})
    assert tuple(ChunkingPipeline.from_spec(spec).chunk(document))


@pytest.mark.asyncio
async def test_async_stream_matches_sync() -> None:
    document = Document(text="alpha beta gamma delta")
    instance = pipeline(2, 0)
    sync = tuple(instance.chunk(document))
    async_values = tuple([chunk async for chunk in instance.achunk(document)])
    assert async_values == sync
