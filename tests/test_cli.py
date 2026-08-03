from __future__ import annotations

import json

from chunkkit.cli import app


def test_preview_cli(capsys) -> None:  # type: ignore[no-untyped-def]
    assert app(["preview", "--text", "one two three", "--chunk-size", "2", "--overlap", "0"]) == 0
    first = json.loads(capsys.readouterr().out.splitlines()[0])
    assert first["text"] == "one two"


def test_config_validate_cli(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "pipeline.json"
    path.write_text('{"chunker":"recursive","chunk_size":8,"overlap":1}', encoding="utf-8")
    assert app(["config", "validate", str(path)]) == 0
    assert capsys.readouterr().out.strip() == "valid"


def test_plugin_list_cli(capsys) -> None:  # type: ignore[no-untyped-def]
    assert app(["plugin", "list", "--group", "chunkers"]) == 0
    assert "chunkkit.chunkers" in capsys.readouterr().out
