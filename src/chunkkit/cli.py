"""Dependency-free command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from . import __version__
from .adapters import write_chunks_jsonl
from .context import ContextPacker
from .models import Chunk, Document, ModelTarget, PipelineSpec, ScoredChunk
from .parsers import parse_path
from .pipeline import ChunkingPipeline
from .plugins import PLUGIN_GROUPS, PluginManager
from .scaffold import scaffold_plugin


def _read_spec(path: str | None, args: argparse.Namespace) -> PipelineSpec:
    if path:
        return PipelineSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
    return PipelineSpec(
        chunker=getattr(args, "chunker", "recursive"),
        chunk_size=getattr(args, "chunk_size", 512),
        overlap=getattr(args, "overlap", 32),
        target=ModelTarget(
            tokenizer=getattr(args, "tokenizer", "unicode"),
            max_input_tokens=getattr(args, "max_input_tokens", 8192),
            reserved_tokens=getattr(args, "reserved_tokens", 1024),
        ),
    )


async def _preview(args: argparse.Namespace) -> int:
    document = (
        await parse_path(args.path, tenant_id=args.tenant)
        if args.path
        else Document(text=args.text or "", tenant_id=args.tenant)
    )
    pipeline = ChunkingPipeline.from_spec(_read_spec(args.config, args))
    chunks = tuple(pipeline.chunk(document))
    if args.output:
        write_chunks_jsonl(chunks, args.output)
    else:
        for chunk in chunks:
            print(chunk.model_dump_json())
    return 0


def _pack(args: argparse.Namespace) -> int:
    chunks = [
        Chunk.model_validate_json(line)
        for line in Path(args.input).read_text().splitlines()
        if line
    ]
    target = ModelTarget(
        tokenizer=args.tokenizer,
        max_input_tokens=args.max_input_tokens,
        reserved_tokens=args.reserved_tokens,
    )
    bundle = ContextPacker(target).assemble([ScoredChunk(chunk=chunk) for chunk in chunks])
    print(bundle.model_dump_json(indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    spec = PipelineSpec.model_validate_json(Path(args.path).read_text(encoding="utf-8"))
    ChunkingPipeline.from_spec(spec)
    print("valid")
    return 0


def _plugins(args: argparse.Namespace) -> int:
    manager = PluginManager()
    groups = [f"chunkkit.{args.group}"] if args.group else list(PLUGIN_GROUPS)
    value = {group: sorted(manager.discover(group)) for group in groups}
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def _scaffold(args: argparse.Namespace) -> int:
    print(scaffold_plugin(args.name, args.destination, group=args.group))
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        from .errors import missing_extra

        raise missing_extra("ChunkKit server", "server", "uvicorn") from exc
    uvicorn.run("chunkkit.server:create_app", host=args.host, port=args.port, factory=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chunkkit", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    preview = subcommands.add_parser("preview", help="Chunk text or a local file")
    source = preview.add_mutually_exclusive_group(required=True)
    source.add_argument("--path")
    source.add_argument("--text")
    preview.add_argument("--config")
    preview.add_argument("--chunker", default="recursive")
    preview.add_argument("--chunk-size", type=int, default=512)
    preview.add_argument("--overlap", type=int, default=32)
    preview.add_argument("--tokenizer", default="unicode")
    preview.add_argument("--max-input-tokens", type=int, default=8192)
    preview.add_argument("--reserved-tokens", type=int, default=1024)
    preview.add_argument("--tenant", default="default")
    preview.add_argument("--output")
    preview.set_defaults(handler=lambda value: asyncio.run(_preview(value)))

    pack = subcommands.add_parser("pack", help="Pack Chunk JSONL into a model context")
    pack.add_argument("input")
    pack.add_argument("--tokenizer", default="unicode")
    pack.add_argument("--max-input-tokens", type=int, default=8192)
    pack.add_argument("--reserved-tokens", type=int, default=1024)
    pack.set_defaults(handler=_pack)

    config = subcommands.add_parser("config", help="Validate pipeline configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate")
    validate.add_argument("path")
    validate.set_defaults(handler=_validate)

    plugin = subcommands.add_parser("plugin", help="Discover or scaffold plugins")
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    plugin_list = plugin_sub.add_parser("list")
    plugin_list.add_argument("--group")
    plugin_list.set_defaults(handler=_plugins)
    scaffold = plugin_sub.add_parser("scaffold")
    scaffold.add_argument("name")
    scaffold.add_argument("--group", default="chunkers")
    scaffold.add_argument("--destination", default=".")
    scaffold.set_defaults(handler=_scaffold)

    models = subcommands.add_parser("models", help="Inspect model budgeting")
    models.add_argument("--tokenizer", default="unicode")
    models.add_argument("--max-input-tokens", type=int, required=True)
    models.add_argument("--reserved-tokens", type=int, default=1024)
    models.set_defaults(
        handler=lambda value: (
            print(
                ModelTarget(
                    tokenizer=value.tokenizer,
                    max_input_tokens=value.max_input_tokens,
                    reserved_tokens=value.reserved_tokens,
                ).model_dump_json(indent=2)
            )
            or 0
        )
    )

    serve = subcommands.add_parser("serve", help="Run the optional FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=_serve)
    return parser


def app(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        result: Any = args.handler(args)
        return int(result or 0)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"chunkkit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(app())
