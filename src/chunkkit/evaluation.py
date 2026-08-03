"""Offline chunk and retrieval evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .models import Document, Metric, RunReport
from .pipeline import ChunkingPipeline
from .vectorstores import LocalVectorIndex


@dataclass(frozen=True, slots=True)
class QueryCase:
    query: str
    relevant_source_uris: tuple[str, ...]
    answers: tuple[str, ...] = ()


def load_dataset(path: str | Path) -> tuple[QueryCase, ...]:
    cases: list[QueryCase] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if "query" not in value or "relevant_source_uris" not in value:
                raise ValueError(f"Dataset line {line_number} lacks query or relevant_source_uris")
            cases.append(
                QueryCase(
                    query=str(value["query"]),
                    relevant_source_uris=tuple(map(str, value["relevant_source_uris"])),
                    answers=tuple(map(str, value.get("answers", ()))),
                )
            )
    return tuple(cases)


async def evaluate_pipeline(
    name: str,
    pipeline: ChunkingPipeline,
    documents: list[Document],
    cases: tuple[QueryCase, ...],
    *,
    k: int = 5,
) -> RunReport:
    started = datetime.now(UTC)
    chunks = tuple(pipeline.chunk_many(documents))
    index = LocalVectorIndex()
    await index.upsert(chunks)
    hits = 0
    reciprocal_ranks: list[float] = []
    discounted_gains: list[float] = []
    for case in cases:
        results = await index.retrieve(
            case.query, k, tenant_id=documents[0].tenant_id if documents else "default"
        )
        relevant = set(case.relevant_source_uris)
        ranks = [
            position
            for position, item in enumerate(results, start=1)
            if item.chunk.source_uri in relevant
        ]
        if ranks:
            hits += 1
            reciprocal_ranks.append(1.0 / ranks[0])
            discounted_gains.append(sum(1.0 / math.log2(rank + 1) for rank in ranks))
        else:
            reciprocal_ranks.append(0.0)
            discounted_gains.append(0.0)

    total_tokens = sum(next(iter(chunk.token_counts.values()), 0) for chunk in chunks)
    denominator = len(cases) or 1
    metrics = (
        Metric(name="chunk_count", value=float(len(chunks))),
        Metric(name="mean_chunk_tokens", value=total_tokens / max(len(chunks), 1)),
        Metric(name=f"hit_rate@{k}", value=hits / denominator),
        Metric(name="mrr", value=sum(reciprocal_ranks) / denominator),
        Metric(name=f"dcg@{k}", value=sum(discounted_gains) / denominator),
        Metric(
            name="token_budget_compliance",
            value=float(
                all(
                    next(iter(chunk.token_counts.values()), 0) <= pipeline.budget.max_tokens
                    for chunk in chunks
                )
            ),
        ),
    )
    return RunReport(
        experiment=name,
        started_at=started,
        finished_at=datetime.now(UTC),
        metrics=metrics,
        metadata={"pipeline": pipeline.spec.model_dump(mode="json"), "queries": len(cases)},
    )


def write_html_report(report: RunReport, path: str | Path) -> None:
    rows = "".join(
        f"<tr><th>{metric.name}</th><td>{metric.value:.6g}</td></tr>" for metric in report.metrics
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>ChunkKit report</title>"
        "<style>body{font-family:system-ui;max-width:900px;margin:3rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.5rem;text-align:left}"
        "</style></head><body>"
        f"<h1>{report.experiment}</h1><table>{rows}</table></body></html>"
    )
    Path(path).write_text(html, encoding="utf-8")
