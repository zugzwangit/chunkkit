"""A complete local-files connector used by the offline demo."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from ..models import ChangeEvent, Checkpoint, RawArtifact, UpsertArtifact
from ..parsers import EXTENSION_MIME_TYPES, parser_for


class FilesystemConnector:
    name = "filesystem"

    def __init__(
        self,
        root: str | Path,
        *,
        tenant_id: str = "default",
        patterns: Iterable[str] = ("**/*",),
        max_bytes: int = 25 * 1024 * 1024,
    ) -> None:
        self.root = Path(root).resolve()
        self.tenant_id = tenant_id
        self.patterns = tuple(patterns)
        self.max_bytes = max_bytes

    async def sync(self, cursor: str | None = None) -> AsyncIterator[ChangeEvent]:
        if not self.root.is_dir():
            raise ValueError(f"Filesystem source is not a directory: {self.root}")
        last_revision = int(cursor or 0)
        newest_revision = last_revision
        paths = sorted(
            {path.resolve() for pattern in self.patterns for path in self.root.glob(pattern)}
        )
        for path in paths:
            if not path.is_file() or not path.is_relative_to(self.root):
                continue
            stat = path.stat()
            revision = stat.st_mtime_ns
            newest_revision = max(newest_revision, revision)
            if revision <= last_revision or stat.st_size > self.max_bytes:
                continue
            content = path.read_bytes()
            mime_type = EXTENSION_MIME_TYPES.get(path.suffix.lower(), "text/plain")
            artifact = RawArtifact(
                source_uri=path.as_uri(),
                revision=str(revision),
                mime_type=mime_type,
                checksum=hashlib.sha256(content).hexdigest(),
                metadata={
                    "filename": path.name,
                    "relative_path": path.relative_to(self.root).as_posix(),
                },
            )
            document = await parser_for(mime_type).parse(artifact, content)
            document = document.model_copy(update={"tenant_id": self.tenant_id})
            yield UpsertArtifact(artifact=artifact, document=document, cursor=str(revision))
        yield Checkpoint(cursor=str(newest_revision))
