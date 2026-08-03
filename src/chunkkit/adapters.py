"""Stable JSONL and optional ecosystem conversions."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from .errors import missing_extra
from .models import Chunk, Document


def write_chunks_jsonl(chunks: Iterable[Chunk], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(chunk.model_dump_json())
            stream.write("\n")


def read_chunks_jsonl(path: str | Path) -> Iterator[Chunk]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield Chunk.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(f"Invalid Chunk JSON on line {line_number}: {exc}") from exc


def to_langchain(chunk: Chunk) -> Any:
    try:
        from langchain_core.documents import Document as LangChainDocument
    except ImportError as exc:  # pragma: no cover
        raise missing_extra("LangChain conversion", "langchain", "langchain-core") from exc
    metadata = chunk.model_dump(mode="json", exclude={"text"})
    return LangChainDocument(page_content=chunk.text, metadata=metadata)


def from_langchain(value: Any, *, source_uri: str = "langchain://document") -> Document:
    return Document(text=value.page_content, source_uri=source_uri, metadata=dict(value.metadata))


def to_llamaindex(chunk: Chunk) -> Any:
    try:
        from llama_index.core.schema import TextNode
    except ImportError as exc:  # pragma: no cover
        raise missing_extra("LlamaIndex conversion", "llamaindex", "llama-index-core") from exc
    return TextNode(
        id_=chunk.id,
        text=chunk.text,
        metadata=chunk.model_dump(mode="json", exclude={"text", "id"}),
    )


def to_haystack(chunk: Chunk) -> Any:
    try:
        from haystack import Document as HaystackDocument
    except ImportError as exc:  # pragma: no cover
        raise missing_extra("Haystack conversion", "haystack", "haystack-ai") from exc
    return HaystackDocument(
        id=chunk.id,
        content=chunk.text,
        meta=chunk.model_dump(mode="json", exclude={"text", "id"}),
    )


def chunk_schema() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(Chunk.model_json_schema())))
