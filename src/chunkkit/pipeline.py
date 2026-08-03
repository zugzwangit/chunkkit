"""Composable chunking pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from hashlib import sha256

from .chunkers import BUILTIN_CHUNKERS
from .errors import ConfigurationError, IncompleteAclError, TokenBudgetError
from .models import (
    AclResolution,
    Chunk,
    ChunkEdge,
    ChunkGraph,
    Document,
    PipelineSpec,
    SourceSpan,
    TokenBudget,
    Visibility,
)
from .plugins import PluginManager
from .protocols import Chunker, Tokenizer
from .tokenizers import create_tokenizer


class ChunkingPipeline:
    def __init__(self, spec: PipelineSpec, chunker: Chunker, tokenizer: Tokenizer) -> None:
        self.spec = spec
        self.chunker = chunker
        self.tokenizer = tokenizer
        self.recipe_hash = sha256(spec.model_dump_json().encode()).hexdigest()

    @classmethod
    def from_spec(
        cls, spec: PipelineSpec, *, plugin_manager: PluginManager | None = None
    ) -> ChunkingPipeline:
        chunker = BUILTIN_CHUNKERS.get(spec.chunker)
        if chunker is None:
            manager = plugin_manager or PluginManager()
            factory = manager.load("chunkkit.chunkers", spec.chunker)
            chunker = factory(**spec.chunker_options) if callable(factory) else factory
        tokenizer_name = spec.tokenizer or spec.target.tokenizer
        try:
            tokenizer = create_tokenizer(tokenizer_name, model=spec.target.model)
        except ConfigurationError:
            manager = plugin_manager or PluginManager()
            factory = manager.load("chunkkit.tokenizers", tokenizer_name)
            tokenizer = factory() if callable(factory) else factory
        return cls(spec, chunker, tokenizer)

    @property
    def budget(self) -> TokenBudget:
        return TokenBudget(
            tokenizer=self.tokenizer.name,
            max_tokens=self.spec.chunk_size or self.spec.target.available_tokens,
            overlap_tokens=self.spec.overlap,
        )

    def _validate_acl(self, document: Document) -> None:
        protected = document.acl.visibility == Visibility.RESTRICTED
        incomplete = document.acl.resolution == AclResolution.INCOMPLETE
        if protected and incomplete and not self.spec.allow_incomplete_acl:
            raise IncompleteAclError(
                f"Refusing to chunk protected document '{document.source_uri}' because its ACL "
                "is incomplete. Configure an ACL mapper or explicitly enable allow_incomplete_acl "
                "for an isolated experiment."
            )

    def chunk(self, document: Document) -> Iterable[Chunk]:
        self._validate_acl(document)
        drafts = list(self.chunker.split(document, self.budget, self.tokenizer))
        identities: list[str] = []
        normalized: list[tuple[int, int, dict[str, object]]] = []
        for ordinal, (start, end, metadata) in enumerate(drafts):
            text = document.text[start:end]
            if not text:
                continue
            token_count = self.tokenizer.count(text)
            if token_count > self.budget.max_tokens:
                raise TokenBudgetError(
                    f"Chunker '{self.chunker.name}' emitted {token_count} tokens for a "
                    f"{self.budget.max_tokens}-token budget"
                )
            payload = (
                f"{document.stable_id}\0{document.revision}\0{start}:{end}\0"
                f"{self.recipe_hash}\0{ordinal}"
            )
            identities.append(sha256(payload.encode()).hexdigest())
            normalized.append((start, end, dict(metadata)))

        for ordinal, ((start, end, metadata), chunk_id) in enumerate(
            zip(normalized, identities, strict=True)
        ):
            text = document.text[start:end]
            yield Chunk(
                id=chunk_id,
                document_id=document.stable_id,
                text=text,
                source_uri=document.source_uri,
                revision=document.revision,
                tenant_id=document.tenant_id,
                spans=(SourceSpan(start=start, end=end),),
                token_counts={self.tokenizer.name: self.tokenizer.count(text)},
                recipe_hash=self.recipe_hash,
                ordinal=ordinal,
                previous_id=identities[ordinal - 1] if ordinal else None,
                next_id=identities[ordinal + 1] if ordinal + 1 < len(identities) else None,
                metadata={**document.metadata, **metadata, "chunker": self.chunker.name},
                acl=document.acl,
            )

    async def achunk(self, document: Document) -> AsyncIterator[Chunk]:
        for chunk in self.chunk(document):
            yield chunk

    def chunk_many(self, documents: Iterable[Document]) -> Iterable[Chunk]:
        for document in documents:
            yield from self.chunk(document)

    def graph(self, document: Document) -> ChunkGraph:
        chunks = tuple(self.chunk(document))
        edges: list[ChunkEdge] = []
        for chunk in chunks:
            if chunk.previous_id:
                edges.append(
                    ChunkEdge(source=chunk.id, target=chunk.previous_id, relation="previous")
                )
            if chunk.next_id:
                edges.append(ChunkEdge(source=chunk.id, target=chunk.next_id, relation="next"))
        return ChunkGraph(chunks=chunks, edges=tuple(edges))
