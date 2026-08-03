from __future__ import annotations

import pytest

from chunkkit import PluginManifest
from chunkkit.plugins import PluginManager
from chunkkit.scaffold import scaffold_plugin
from chunkkit.storage import (
    EnvironmentSecretResolver,
    InMemoryJobBroker,
    LocalArtifactStore,
    SqliteStore,
)


@pytest.mark.asyncio
async def test_sqlite_and_artifact_stores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = SqliteStore(tmp_path / "state.db")
    await store.put("pipelines", "one", {"value": 1})
    assert await store.get("pipelines", "one") == {"value": 1}
    assert await store.load("tenant", "source") is None
    await store.save("tenant", "source", "cursor")
    assert await store.load("tenant", "source") == "cursor"

    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    await artifacts.put("tenant/report.json", b"{}")
    assert await artifacts.get("tenant/report.json") == b"{}"
    with pytest.raises(ValueError):
        await artifacts.put("../escape", b"bad")


@pytest.mark.asyncio
async def test_secret_and_job_backends(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("CHUNKKIT_TEST_SECRET", "value")
    assert await EnvironmentSecretResolver().resolve("env://CHUNKKIT_TEST_SECRET") == "value"
    broker = InMemoryJobBroker()
    job_id = await broker.enqueue("tenant", {"operation": "chunk"})
    assert (await broker.dequeue())[:2] == (job_id, "tenant")


def test_plugin_manifest_and_scaffold(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = PluginManifest(name="demo", version="1.0", capabilities=("chunkers",))
    assert manifest.api_version == "1"
    root = scaffold_plugin("Acme Semantic", tmp_path)
    assert (root / "pyproject.toml").exists()
    assert "chunkkit.chunkers" in (root / "pyproject.toml").read_text(encoding="utf-8")


def test_plugin_manager_rejects_unknown_group() -> None:
    with pytest.raises(Exception, match="Unknown plugin group"):
        PluginManager().discover("not.a.group")
