"""A custom chunker can be passed directly or published as an entry-point plugin."""

from chunkkit import ChunkingPipeline, Document, ModelTarget, PipelineSpec
from chunkkit.tokenizers import UnicodeTokenizer


class WholeDocumentChunker:
    name = "whole-document"

    def split(self, document, budget, tokenizer):  # type: ignore[no-untyped-def]
        if tokenizer.count(document.text) > budget.max_tokens:
            raise ValueError("document is too large for this intentionally simple chunker")
        yield 0, len(document.text), {"custom": True}


spec = PipelineSpec(
    chunk_size=64,
    overlap=0,
    target=ModelTarget(max_input_tokens=128, reserved_tokens=32),
)
pipeline = ChunkingPipeline(spec, WholeDocumentChunker(), UnicodeTokenizer())
print(tuple(pipeline.chunk(Document(text="Custom integrations need no framework adapter."))))
