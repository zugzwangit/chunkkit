# Architecture

ChunkKit separates source acquisition from content normalization and chunking. The canonical data
model is the boundary between optional integrations.

```text
ChangeEvent -> RawArtifact -> Parser -> Document -> Chunker -> ChunkGraph
                                                        |       |
                                                        |       +-> VectorStore
                                                        +-> evaluation
Retriever -> Reranker -> ContextPacker -> ContextBundle -> external Generator
```

`Document` retains source identity, revision, structure, discarded spans, metadata, tenant, and ACL.
`Chunk` adds stable IDs, exact source spans, token counts, recipe identity, and graph relationships.
The pipeline commits no connector checkpoint itself: an orchestrator must persist a checkpoint only
after all prior events have reached their sinks.

The core package imports only Pydantic. Optional modules may import their dependency inside the
constructor or method that needs it and must produce an actionable extras error.

## Stability

Schemas include `schema_version`. Chunk IDs include the complete serialized `PipelineSpec`, so a
recipe change intentionally creates new IDs. Plugin APIs are structural protocols and entry-point
groups rather than inheritance hierarchies.
