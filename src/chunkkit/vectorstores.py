"""Offline exact retrieval and optional production vector-store adapters."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from hashlib import blake2b
from typing import Any

from .errors import missing_extra
from .models import Chunk, Principal, ScoredChunk, Visibility


class HashingEmbedder:
    """Deterministic offline embedder for demos and contract tests, not production ranking."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def _one(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        for word in text.casefold().split():
            digest = blake2b(word.encode(), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            values[index] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return tuple(value / norm for value in values)

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._one(text) for text in texts)


def _visible(chunk: Chunk, principals: Sequence[Principal]) -> bool:
    if chunk.acl.visibility == Visibility.PUBLIC:
        return True
    allowed = {principal.key for principal in chunk.acl.allow}
    caller = {principal.key for principal in principals}
    denied = {principal.key for principal in chunk.acl.deny}
    return bool(allowed & caller) and not bool(denied & caller)


class LocalVectorIndex:
    """Exact cosine index for offline evaluation."""

    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._items: dict[str, tuple[Chunk, tuple[float, ...]]] = {}

    async def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]] | None = None
    ) -> None:
        materialized = vectors or await self.embedder.embed([chunk.text for chunk in chunks])
        if len(materialized) != len(chunks):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, materialized, strict=True):
            self._items[chunk.id] = (chunk, tuple(float(value) for value in vector))

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._items.pop(chunk_id, None)

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        *,
        tenant_id: str = "default",
        principals: Sequence[Principal] = (),
    ) -> Sequence[ScoredChunk]:
        query_vector = tuple((await self.embedder.embed([query]))[0])
        scored: list[ScoredChunk] = []
        for chunk, vector in self._items.values():
            if chunk.tenant_id != tenant_id or not _visible(chunk, principals):
                continue
            score = sum(left * right for left, right in zip(query_vector, vector, strict=False))
            scored.append(ScoredChunk(chunk=chunk, score=score))
        return tuple(sorted(scored, key=lambda item: item.score, reverse=True)[:limit])


class QdrantVectorStore:
    def __init__(self, client: Any, *, collection_prefix: str = "chunkkit") -> None:
        try:
            from qdrant_client import models as qmodels
        except ImportError as exc:  # pragma: no cover
            raise missing_extra("QdrantVectorStore", "qdrant", "qdrant-client") from exc
        self.client = client
        self.collection_prefix = collection_prefix
        self._models = qmodels

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        by_tenant: dict[str, list[tuple[Chunk, Sequence[float]]]] = {}
        for chunk, vector in zip(chunks, vectors, strict=True):
            by_tenant.setdefault(chunk.tenant_id, []).append((chunk, vector))
        for tenant, items in by_tenant.items():
            collection = f"{self.collection_prefix}_{tenant}"
            points = [
                self._models.PointStruct(
                    id=chunk.id,
                    vector=list(vector),
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, vector in items
            ]
            await self.client.upsert(collection_name=collection, points=points)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        raise ValueError(
            "Qdrant deletion requires a tenant-specific collection; use delete_for_tenant"
        )

    async def delete_for_tenant(self, tenant_id: str, chunk_ids: Sequence[str]) -> None:
        await self.client.delete(
            collection_name=f"{self.collection_prefix}_{tenant_id}",
            points_selector=self._models.PointIdsList(points=list(chunk_ids)),
        )


class PineconeVectorStore:
    def __init__(self, index: Any) -> None:
        try:
            import pinecone  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise missing_extra("PineconeVectorStore", "pinecone", "pinecone") from exc
        self.index = index

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        by_tenant: dict[str, list[dict[str, object]]] = {}
        for chunk, vector in zip(chunks, vectors, strict=True):
            by_tenant.setdefault(chunk.tenant_id, []).append(
                {"id": chunk.id, "values": list(vector), "metadata": chunk.model_dump(mode="json")}
            )
        for tenant, values in by_tenant.items():
            self.index.upsert(vectors=values, namespace=tenant)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        raise ValueError("Pinecone deletion requires a tenant namespace; use delete_for_tenant")

    async def delete_for_tenant(self, tenant_id: str, chunk_ids: Sequence[str]) -> None:
        self.index.delete(ids=list(chunk_ids), namespace=tenant_id)


class PgVectorStore:
    """Async SQLAlchemy/pgvector writer with mandatory tenant predicates."""

    def __init__(self, engine: Any, *, table: str = "chunkkit_chunks") -> None:
        try:
            from sqlalchemy import bindparam, text
        except ImportError as exc:  # pragma: no cover
            raise missing_extra("PgVectorStore", "pgvector", "sqlalchemy") from exc
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("pgvector table must be a simple SQL identifier")
        self.engine = engine
        self.table = table
        self._text = text
        self._bindparam = bindparam

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        statement = self._text(
            f"INSERT INTO {self.table} (id, tenant_id, embedding, payload) "
            "VALUES (:id, :tenant_id, CAST(:embedding AS vector), CAST(:payload AS jsonb)) "
            "ON CONFLICT (id, tenant_id) DO UPDATE SET "
            "embedding=excluded.embedding, payload=excluded.payload"
        )
        rows = [
            {
                "id": chunk.id,
                "tenant_id": chunk.tenant_id,
                "embedding": "[" + ",".join(map(str, vector)) + "]",
                "payload": chunk.model_dump_json(),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        async with self.engine.begin() as connection:
            await connection.execute(statement, rows)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        raise ValueError("pgvector deletion requires tenant_id; use delete_for_tenant")

    async def delete_for_tenant(self, tenant_id: str, chunk_ids: Sequence[str]) -> None:
        statement = self._text(
            f"DELETE FROM {self.table} WHERE tenant_id=:tenant_id AND id IN :chunk_ids"
        ).bindparams(self._bindparam("chunk_ids", expanding=True))
        async with self.engine.begin() as connection:
            await connection.execute(
                statement, {"tenant_id": tenant_id, "chunk_ids": list(chunk_ids)}
            )
