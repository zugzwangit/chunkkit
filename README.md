# ChunkKit

[![CI](https://github.com/zugzwangit/chunkkit/actions/workflows/ci.yml/badge.svg)](https://github.com/zugzwangit/chunkkit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

ChunkKit is a framework-neutral, model-aware chunking toolkit for RAG, retrieval, and LLM
inference pipelines. The core is deliberately small: canonical data contracts, deterministic
chunkers, exact token budgets, context packing, and a public plugin SDK. Cloud connectors,
model providers, vector databases, evaluation tools, and the HTTP service are optional.

> **Project status:** early alpha. The core schema is versioned, but public APIs may evolve before
> 1.0 under the compatibility policy in [GOVERNANCE.md](GOVERNANCE.md).

## Why ChunkKit?

- Use it directly without LangChain, LlamaIndex, or Haystack.
- Preserve source spans, revisions, deterministic IDs, metadata, and ACLs.
- Size chunks against explicit tokenizer/model budgets instead of character estimates.
- Swap chunkers, connectors, tokenizers, retrievers, stores, and evaluators through protocols.
- Run fully offline. ChunkKit makes no network calls unless a configured plugin does so.
- Compare strategies with a deterministic local embedder and exact retrieval index.

## Five-minute quickstart

```bash
python -m pip install chunkkit
```

```python
from chunkkit import Document, ModelTarget, PipelineSpec, ChunkingPipeline

document = Document(
    text="""# ChunkKit

ChunkKit produces deterministic, traceable chunks for arbitrary LLM pipelines.

## Plugins

Every external integration implements a small public protocol.
""",
    source_uri="memory://readme",
)

pipeline = ChunkingPipeline.from_spec(
    PipelineSpec(
        chunker="recursive",
        chunk_size=32,
        overlap=4,
        target=ModelTarget(
            tokenizer="unicode",
            max_input_tokens=128,
            reserved_tokens=32,
        ),
    )
)

for chunk in pipeline.chunk(document):
    print(chunk.id, chunk.text, chunk.spans)
```

Async streaming uses the same pipeline:

```python
async for chunk in pipeline.achunk(document):
    await your_llm_pipeline.send(chunk)
```

Preview a file from the CLI:

```bash
chunkkit preview --path README.md --chunker markdown --chunk-size 256 --overlap 32
```

## Architecture

```text
SourceConnector -> Parser -> Document -> ChunkingPipeline -> ChunkGraph
                                                        |-> VectorStore
Retriever -> optional Reranker -> ContextPacker -> ContextBundle -> your LLM
```

The canonical `Document`, `Chunk`, `ChunkGraph`, and `ContextBundle` types are stable integration
points. All records serialize to versioned JSON. `ChunkingPipeline` accepts custom objects that
implement the public protocols in `chunkkit.protocols`.

Built-in deterministic strategies include token, sliding-window, recursive, sentence, paragraph,
Markdown, code-structure fallback, and adaptive selection. Provider-exact tokenizers such as
`tiktoken` are optional; the dependency-free `unicode` tokenizer is intended for offline workflows
and testing.

## Optional capabilities

Install only what the deployment needs:

```bash
pip install 'chunkkit[server]'
pip install 'chunkkit[documents,code,eval]'
pip install 'chunkkit[atlassian,qdrant,openai]'
pip install 'chunkkit[langchain,llamaindex,haystack]'
```

Connector classes are available for Confluence, Jira, Notion, Google Drive, Microsoft Graph,
Slack, GitHub, ServiceNow, and Zendesk. Their base normalizer is intentionally configurable:
enterprise installations should provide field mappings and an `AclMapper` matching local policy.
The local filesystem connector is complete and powers the offline demo.

The optional API starts with:

```bash
pip install 'chunkkit[server]'
chunkkit serve
curl -H 'Authorization: Bearer dev' http://127.0.0.1:8000/health/ready
```

The `dev` token is a local demonstration default. Configure `CHUNKKIT_API_KEYS` before any shared
deployment. See [docs/security-model.md](docs/security-model.md).

## Writing a plugin

Generate a standalone project:

```bash
chunkkit plugin scaffold acme_semantic --group chunkers --destination ./plugins
```

Plugins use standard Python entry points and can be distributed independently. See
[docs/plugins.md](docs/plugins.md) and [examples/custom_chunker.py](examples/custom_chunker.py).

## Repository map

- `src/chunkkit`: core library and optional adapter modules.
- `tests`: core, security, connector, CLI, and service contract tests.
- `examples`: framework-free integration examples.
- `docs`: architecture, plugins, security, and compatibility.
- `.github`: CI, issue templates, and release automation.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup. Report vulnerabilities through
GitHub private vulnerability reporting as described in [SECURITY.md](SECURITY.md); do not open a
public issue for an undisclosed vulnerability.

ChunkKit is licensed under the [Apache License 2.0](LICENSE) and does not collect telemetry.
