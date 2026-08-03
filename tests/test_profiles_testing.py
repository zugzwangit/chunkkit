from __future__ import annotations

import json

import pytest

from chunkkit import ChunkingPipeline, Document, ModelProfile, ModelTarget, PipelineSpec
from chunkkit.chunkers import FixedTokenChunker
from chunkkit.errors import ConfigurationError
from chunkkit.profiles import ModelProfileRegistry
from chunkkit.testing import assert_chunker_contract, assert_connector_contract
from chunkkit.tokenizers import CharacterTokenizer, UnicodeTokenizer, create_tokenizer


def test_model_profile_registry_requires_explicit_profile(tmp_path) -> None:  # type: ignore[no-untyped-def]
    profile = ModelProfile(
        provider="local",
        model="demo",
        tokenizer="unicode",
        context_window=4096,
        max_input_tokens=4096,
        max_output_tokens=512,
    )
    path = tmp_path / "models.json"
    path.write_text(json.dumps([profile.model_dump(mode="json")]), encoding="utf-8")
    registry = ModelProfileRegistry.from_json(path)
    assert registry.resolve("local", "demo") == profile
    assert registry.target("local", "demo", reserved_tokens=96).available_tokens == 3936
    with pytest.raises(ConfigurationError, match="will not guess"):
        registry.resolve("unknown", "model")


def test_tokenizer_and_chunker_contract() -> None:
    assert CharacterTokenizer().count("abc") == 3
    assert create_tokenizer("unicode").count("hello, world") == 3
    assert_chunker_contract(FixedTokenChunker(), UnicodeTokenizer())


@pytest.mark.asyncio
async def test_connector_contract() -> None:
    class Connector:
        name = "test"

        async def sync(self, cursor=None):  # type: ignore[no-untyped-def]
            from chunkkit import Checkpoint

            yield Checkpoint(cursor=cursor or "done")

    await assert_connector_contract(Connector())


def test_pipeline_plugin_can_be_passed_directly() -> None:
    instance = ChunkingPipeline(
        PipelineSpec(
            chunk_size=4,
            overlap=0,
            target=ModelTarget(max_input_tokens=16, reserved_tokens=1, safety_margin_tokens=1),
        ),
        FixedTokenChunker(),
        UnicodeTokenizer(),
    )
    assert tuple(instance.chunk(Document(text="one two three")))
