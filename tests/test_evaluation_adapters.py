from __future__ import annotations

import pytest

from chunkkit import ChunkingPipeline, Document, ModelTarget, PipelineSpec, Principal
from chunkkit.adapters import read_chunks_jsonl, write_chunks_jsonl
from chunkkit.evaluation import QueryCase, evaluate_pipeline, write_html_report
from chunkkit.models import Acl, Visibility
from chunkkit.vectorstores import LocalVectorIndex


def make_pipeline() -> ChunkingPipeline:
    return ChunkingPipeline.from_spec(
        PipelineSpec(
            chunk_size=8,
            overlap=1,
            target=ModelTarget(max_input_tokens=32, reserved_tokens=2, safety_margin_tokens=1),
        )
    )


def test_jsonl_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    chunks = tuple(make_pipeline().chunk(Document(text="alpha beta gamma", source_uri="x://one")))
    path = tmp_path / "chunks.jsonl"
    write_chunks_jsonl(chunks, path)
    assert tuple(read_chunks_jsonl(path)) == chunks


@pytest.mark.asyncio
async def test_local_index_enforces_tenant_and_acl() -> None:
    public = next(make_pipeline().chunk(Document(text="public answer", source_uri="x://public")))
    private = next(
        make_pipeline().chunk(
            Document(
                text="private answer",
                source_uri="x://private",
                acl=Acl(
                    visibility=Visibility.RESTRICTED,
                    allow=(Principal(kind="group", identifier="eng"),),
                ),
            )
        )
    )
    index = LocalVectorIndex()
    await index.upsert([public, private])
    anonymous = await index.retrieve("answer", principals=())
    assert {item.chunk.id for item in anonymous} == {public.id}
    authorized = await index.retrieve(
        "answer", principals=(Principal(kind="group", identifier="eng"),)
    )
    assert {item.chunk.id for item in authorized} == {public.id, private.id}


@pytest.mark.asyncio
async def test_offline_evaluation_and_html(tmp_path) -> None:  # type: ignore[no-untyped-def]
    documents = [Document(text="Paris is the capital of France.", source_uri="doc://france")]
    report = await evaluate_pipeline(
        "demo",
        make_pipeline(),
        documents,
        (QueryCase("capital France", ("doc://france",)),),
    )
    assert {metric.name for metric in report.metrics} >= {"mrr", "token_budget_compliance"}
    output = tmp_path / "report.html"
    write_html_report(report, output)
    assert "ChunkKit report" in output.read_text(encoding="utf-8")
