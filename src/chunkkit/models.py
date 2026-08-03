"""Canonical, framework-neutral data contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)


class ElementKind(StrEnum):
    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    COMMENT = "comment"
    ATTACHMENT = "attachment"
    OTHER = "other"


class Visibility(StrEnum):
    PUBLIC = "public"
    RESTRICTED = "restricted"


class AclResolution(StrEnum):
    COMPLETE = "complete"
    MAPPED = "mapped"
    INCOMPLETE = "incomplete"


class Principal(FrozenModel):
    kind: Literal["user", "group", "domain", "service", "public"]
    identifier: str = Field(min_length=1)
    source: str | None = None

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.identifier}"


class Acl(FrozenModel):
    visibility: Visibility = Visibility.PUBLIC
    allow: tuple[Principal, ...] = ()
    deny: tuple[Principal, ...] = ()
    resolution: AclResolution = AclResolution.COMPLETE
    source: str | None = None

    @model_validator(mode="after")
    def restricted_has_policy(self) -> Acl:
        if (
            self.visibility == Visibility.RESTRICTED
            and not self.allow
            and self.resolution != AclResolution.INCOMPLETE
        ):
            raise ValueError("restricted ACLs must contain at least one allowed principal")
        return self


class SourceSpan(FrozenModel):
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    pointer: str | None = None
    page: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_span(self) -> SourceSpan:
        if self.start is None and self.pointer is None:
            raise ValueError("a source span requires offsets or a structured pointer")
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be provided together")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("span end must be greater than or equal to start")
        return self


class Element(FrozenModel):
    kind: ElementKind
    text: str = ""
    span: SourceSpan | None = None
    children: tuple[Element, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawArtifact(FrozenModel):
    schema_version: str = SCHEMA_VERSION
    source_uri: str = Field(min_length=1)
    revision: str = ""
    mime_type: str = "text/plain"
    checksum: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    content_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscardedSpan(FrozenModel):
    span: SourceSpan
    reason: str = Field(min_length=1)


class Document(FrozenModel):
    schema_version: str = SCHEMA_VERSION
    id: str | None = None
    text: str
    source_uri: str = "memory://document"
    revision: str = ""
    mime_type: str = "text/plain"
    tenant_id: str = "default"
    title: str | None = None
    elements: tuple[Element, ...] = ()
    discarded_spans: tuple[DiscardedSpan, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: Acl = Field(default_factory=Acl)

    @property
    def stable_id(self) -> str:
        if self.id:
            return self.id
        payload = f"{self.tenant_id}\0{self.source_uri}\0{self.revision}"
        return sha256(payload.encode()).hexdigest()


class ModelTarget(FrozenModel):
    tokenizer: str = "unicode"
    model: str | None = None
    role: Literal["embedding", "reranking", "generation", "generic"] = "generic"
    max_input_tokens: int = Field(default=8192, gt=0)
    reserved_tokens: int = Field(default=1024, ge=0)
    safety_margin_tokens: int = Field(default=64, ge=0)

    @property
    def available_tokens(self) -> int:
        return self.max_input_tokens - self.reserved_tokens - self.safety_margin_tokens

    @model_validator(mode="after")
    def budget_is_positive(self) -> ModelTarget:
        if self.available_tokens <= 0:
            raise ValueError("reserved and safety-margin tokens exhaust the model input")
        return self


class ModelProfile(FrozenModel):
    provider: str
    model: str
    tokenizer: str
    context_window: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(ge=0)
    modalities: tuple[str, ...] = ("text",)
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenBudget(FrozenModel):
    tokenizer: str
    max_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def overlap_is_smaller(self) -> TokenBudget:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class PipelineSpec(FrozenModel):
    schema_version: str = SCHEMA_VERSION
    chunker: str = "recursive"
    tokenizer: str | None = None
    target: ModelTarget = Field(default_factory=ModelTarget)
    chunk_size: int | None = Field(default=512, gt=0)
    overlap: int = Field(default=32, ge=0)
    chunker_options: dict[str, Any] = Field(default_factory=dict)
    allow_incomplete_acl: bool = False

    @model_validator(mode="after")
    def chunk_size_fits(self) -> PipelineSpec:
        size = self.chunk_size or self.target.available_tokens
        if size > self.target.available_tokens:
            raise ValueError("chunk_size exceeds the available model input budget")
        if self.overlap >= size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class Chunk(FrozenModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    document_id: str
    text: str
    source_uri: str
    revision: str = ""
    tenant_id: str = "default"
    spans: tuple[SourceSpan, ...]
    token_counts: dict[str, int]
    recipe_hash: str
    ordinal: int = Field(ge=0)
    parent_id: str | None = None
    previous_id: str | None = None
    next_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: Acl = Field(default_factory=Acl)


class ChunkEdge(FrozenModel):
    source: str
    target: str
    relation: Literal["parent", "child", "previous", "next", "continuation", "summary"]


class ChunkGraph(FrozenModel):
    chunks: tuple[Chunk, ...]
    edges: tuple[ChunkEdge, ...] = ()


class ScoredChunk(FrozenModel):
    chunk: Chunk
    score: float = 0.0


class DroppedChunk(FrozenModel):
    chunk_id: str
    reason: Literal["duplicate", "budget", "acl", "empty"]
    token_count: int = Field(ge=0)


class Citation(FrozenModel):
    chunk_id: str
    source_uri: str
    spans: tuple[SourceSpan, ...]


class ContextBundle(FrozenModel):
    text: str
    chunks: tuple[Chunk, ...]
    dropped: tuple[DroppedChunk, ...]
    citations: tuple[Citation, ...]
    token_count: int = Field(ge=0)
    max_tokens: int = Field(gt=0)


class ExperimentSpec(FrozenModel):
    name: str
    pipelines: tuple[PipelineSpec, ...]
    dataset_uri: str
    seed: int = 0
    max_cost: float | None = Field(default=None, ge=0)


class Metric(FrozenModel):
    name: str
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunReport(FrozenModel):
    experiment: str
    started_at: datetime
    finished_at: datetime
    metrics: tuple[Metric, ...]
    artifacts: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertArtifact(FrozenModel):
    kind: Literal["upsert"] = "upsert"
    artifact: RawArtifact
    document: Document | None = None
    cursor: str | None = None


class DeleteArtifact(FrozenModel):
    kind: Literal["delete"] = "delete"
    source_uri: str
    revision: str = ""
    cursor: str | None = None


class Checkpoint(FrozenModel):
    kind: Literal["checkpoint"] = "checkpoint"
    cursor: str


ChangeEvent = UpsertArtifact | DeleteArtifact | Checkpoint
