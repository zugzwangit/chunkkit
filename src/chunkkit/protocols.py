"""Runtime-checkable extension protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .models import (
    Acl,
    ChangeEvent,
    Chunk,
    ContextBundle,
    Document,
    ModelProfile,
    Principal,
    RawArtifact,
    RunReport,
    ScoredChunk,
    TokenBudget,
)


@runtime_checkable
class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def spans(self, text: str) -> Sequence[tuple[int, int]]: ...


@runtime_checkable
class Chunker(Protocol):
    name: str

    def split(
        self, document: Document, budget: TokenBudget, tokenizer: Tokenizer
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]: ...


@runtime_checkable
class SourceConnector(Protocol):
    name: str

    def sync(self, cursor: str | None = None) -> AsyncIterator[ChangeEvent]: ...


@runtime_checkable
class Parser(Protocol):
    name: str

    async def parse(self, artifact: RawArtifact, content: bytes) -> Document: ...


@runtime_checkable
class Normalizer(Protocol):
    name: str

    async def normalize(self, document: Document) -> Document: ...


@runtime_checkable
class StrategySelector(Protocol):
    def select(self, document: Document, profile: ModelProfile | None) -> str: ...


@runtime_checkable
class ModelProfileResolver(Protocol):
    async def resolve(self, provider: str, model: str) -> ModelProfile: ...


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, limit: int = 10) -> Sequence[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, chunks: Sequence[ScoredChunk]) -> Sequence[ScoredChunk]: ...


@runtime_checkable
class ContextAssembler(Protocol):
    def assemble(self, chunks: Sequence[ScoredChunk | Chunk]) -> ContextBundle: ...


@runtime_checkable
class Generator(Protocol):
    async def generate(self, prompt: str, context: ContextBundle) -> str: ...


@runtime_checkable
class Judge(Protocol):
    async def judge(
        self, query: str, answer: str, context: ContextBundle
    ) -> Mapping[str, float]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    async def delete(self, chunk_ids: Sequence[str]) -> None: ...


@runtime_checkable
class CheckpointStore(Protocol):
    async def load(self, tenant_id: str, source: str) -> str | None: ...

    async def save(self, tenant_id: str, source: str, cursor: str) -> None: ...


@runtime_checkable
class MetadataStore(Protocol):
    async def put(self, namespace: str, key: str, value: Mapping[str, Any]) -> None: ...

    async def get(self, namespace: str, key: str) -> Mapping[str, Any] | None: ...


@runtime_checkable
class ArtifactStore(Protocol):
    async def put(self, key: str, content: bytes) -> str: ...

    async def get(self, key: str) -> bytes: ...


@runtime_checkable
class JobBroker(Protocol):
    async def enqueue(self, tenant_id: str, payload: Mapping[str, Any]) -> str: ...


@runtime_checkable
class SecretResolver(Protocol):
    async def resolve(self, reference: str) -> str: ...


@runtime_checkable
class IdentityMapper(Protocol):
    def principals(self, claims: Mapping[str, Any]) -> Sequence[Principal]: ...


@runtime_checkable
class AclResolver(Protocol):
    async def resolve(self, artifact: RawArtifact) -> Acl: ...


@runtime_checkable
class Evaluator(Protocol):
    async def evaluate(self, dataset_uri: str) -> RunReport: ...
