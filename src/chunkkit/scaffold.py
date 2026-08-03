"""Generate a standalone third-party ChunkKit plugin project."""

from __future__ import annotations

import re
from pathlib import Path

from .errors import ConfigurationError


def scaffold_plugin(name: str, destination: str | Path, *, group: str = "chunkers") -> Path:
    normalized = re.sub(r"[^a-z0-9_]+", "_", name.casefold().replace("-", "_")).strip("_")
    if not normalized or group not in {
        "chunkers",
        "connectors",
        "parsers",
        "tokenizers",
        "models",
        "vectorstores",
        "evaluators",
        "storage",
    }:
        raise ConfigurationError("invalid plugin name or group")
    root = Path(destination).resolve() / f"chunkkit-{normalized}"
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {root}")
    package = root / "src" / f"chunkkit_{normalized}"
    tests = root / "tests"
    package.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    class_name = "".join(part.title() for part in normalized.split("_")) + "Plugin"
    protocol_method = {
        "chunkers": (
            '    name = "' + normalized + '"\n\n'
            "    def split(self, document, budget, tokenizer):\n"
            '        yield 0, len(document.text), {"plugin": self.name}\n'
        ),
        "connectors": (
            '    name = "' + normalized + '"\n\n'
            "    async def sync(self, cursor=None):\n"
            "        if False:\n            yield cursor\n"
        ),
    }.get(group, '    name = "' + normalized + '"\n')
    (package / "__init__.py").write_text(
        "from chunkkit import PluginManifest\n\n\n"
        f"class {class_name}:\n{protocol_method}\n\n"
        "manifest = PluginManifest(\n"
        f'    name="{normalized}", version="0.1.0", capabilities=("{group}",)\n'
        ")\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
        "[project]\n"
        f'name = "chunkkit-{normalized}"\nversion = "0.1.0"\n'
        'requires-python = ">=3.12"\ndependencies = ["chunkkit>=0.1,<1"]\n\n'
        f'[project.entry-points."chunkkit.{group}"]\n'
        f'{normalized} = "chunkkit_{normalized}:{class_name}"\n\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/chunkkit_'
        f'{normalized}"]\n',
        encoding="utf-8",
    )
    (tests / "test_plugin.py").write_text(
        f"from chunkkit_{normalized} import {class_name}, manifest\n\n\n"
        'def test_manifest():\n    assert manifest.api_version == "1"\n\n\n'
        f"def test_plugin_constructs():\n    assert {class_name}() is not None\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"# chunkkit-{normalized}\n\nGenerated ChunkKit `{group}` plugin.\n",
        encoding="utf-8",
    )
    return root
