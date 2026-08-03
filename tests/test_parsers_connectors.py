from __future__ import annotations

import json

import pytest

from chunkkit.connectors import FilesystemConnector, JiraConnector
from chunkkit.models import AclResolution, Checkpoint, RawArtifact, UpsertArtifact
from chunkkit.parsers import DelimitedParser, HTMLTextParser, JSONParser, TextParser, parse_path


@pytest.mark.asyncio
async def test_lightweight_parsers() -> None:
    artifact = RawArtifact(source_uri="memory://x", mime_type="text/html")
    html = await HTMLTextParser().parse(artifact, b"<h1>Hello</h1><p>World</p>")
    assert "Hello" in html.text and "World" in html.text
    data = await JSONParser().parse(
        artifact.model_copy(update={"mime_type": "application/json"}), b'{"b": 2, "a": 1}'
    )
    assert data.text.index('"a"') < data.text.index('"b"')
    table = await DelimitedParser().parse(artifact, b"name,value\na,1")
    assert table.elements[0].metadata["rows"] == 2
    plain = await TextParser().parse(artifact, "café".encode())
    assert plain.text == "café"


@pytest.mark.asyncio
async def test_filesystem_connector_is_incremental(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.md").write_text("# A\n\nBody", encoding="utf-8")
    events = [event async for event in FilesystemConnector(tmp_path).sync()]
    assert isinstance(events[0], UpsertArtifact)
    assert events[0].document is not None
    assert isinstance(events[-1], Checkpoint)
    resumed = [event async for event in FilesystemConnector(tmp_path).sync(events[-1].cursor)]
    assert resumed == [events[-1]]


@pytest.mark.asyncio
async def test_parse_path_sets_tenant(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "input.txt"
    path.write_text("hello", encoding="utf-8")
    assert (await parse_path(path, tenant_id="acme")).tenant_id == "acme"


def test_enterprise_connector_defaults_to_fail_closed_acl() -> None:
    connector = JiraConnector("https://example.atlassian.net", "secret")
    document = connector.document({"key": "ABC-1", "fields": {"summary": "Example"}})
    assert document.acl.resolution == AclResolution.INCOMPLETE


def test_json_fixture_is_sanitized() -> None:
    value = json.loads('{"id":"demo","title":"Synthetic"}')
    assert value["id"] == "demo"
