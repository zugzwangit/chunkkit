"""Use ChunkKit as one stage inside an arbitrary async LLM pipeline."""

import asyncio

from chunkkit import ChunkingPipeline, ContextPacker, Document, ModelTarget, PipelineSpec
from chunkkit.vectorstores import LocalVectorIndex


async def main() -> None:
    target = ModelTarget(max_input_tokens=128, reserved_tokens=32)
    pipeline = ChunkingPipeline.from_spec(
        PipelineSpec(chunker="recursive", chunk_size=24, overlap=4, target=target)
    )
    document = Document(
        text="ChunkKit can feed any retriever or LLM. Its canonical chunks preserve citations.",
        source_uri="example://framework-neutral",
    )
    chunks = [chunk async for chunk in pipeline.achunk(document)]
    index = LocalVectorIndex()
    await index.upsert(chunks)
    retrieved = await index.retrieve("What can ChunkKit feed?")
    context = ContextPacker(target).assemble(retrieved)
    print(context.text)
    print([citation.source_uri for citation in context.citations])


if __name__ == "__main__":
    asyncio.run(main())
