"""Dependency-free local reference backends."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SqliteStore:
    """Demo metadata and checkpoint store; use a production plugin for distributed use."""

    def __init__(self, path: str | Path = ".chunkkit/state.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY(namespace, key)
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    tenant_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    cursor TEXT NOT NULL,
                    PRIMARY KEY(tenant_id, source)
                );
                """
            )

    async def put(self, namespace: str, key: str, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True)
        await asyncio.to_thread(self._put, namespace, key, payload)

    def _put(self, namespace: str, key: str, payload: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO metadata(namespace,key,value) VALUES(?,?,?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value",
                (namespace, key, payload),
            )

    async def get(self, namespace: str, key: str) -> Mapping[str, Any] | None:
        value = await asyncio.to_thread(self._get, namespace, key)
        return json.loads(value) if value is not None else None

    def _get(self, namespace: str, key: str) -> str | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE namespace=? AND key=?", (namespace, key)
            ).fetchone()
        return str(row[0]) if row else None

    async def load(self, tenant_id: str, source: str) -> str | None:
        return await asyncio.to_thread(self._load_checkpoint, tenant_id, source)

    def _load_checkpoint(self, tenant_id: str, source: str) -> str | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT cursor FROM checkpoints WHERE tenant_id=? AND source=?",
                (tenant_id, source),
            ).fetchone()
        return str(row[0]) if row else None

    async def save(self, tenant_id: str, source: str, cursor: str) -> None:
        await asyncio.to_thread(self._save_checkpoint, tenant_id, source, cursor)

    def _save_checkpoint(self, tenant_id: str, source: str, cursor: str) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO checkpoints(tenant_id,source,cursor) VALUES(?,?,?) "
                "ON CONFLICT(tenant_id,source) DO UPDATE SET cursor=excluded.cursor",
                (tenant_id, source, cursor),
            )


class LocalArtifactStore:
    def __init__(self, root: str | Path = ".chunkkit/artifacts") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        target = (self.root / key).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("artifact key escapes the configured root")
        return target

    async def put(self, key: str, content: bytes) -> str:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, content)
        return target.as_uri()

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)


class EnvironmentSecretResolver:
    async def resolve(self, reference: str) -> str:
        prefix = "env://"
        if not reference.startswith(prefix):
            raise ValueError("the environment resolver accepts only env://NAME references")
        name = reference.removeprefix(prefix)
        try:
            return os.environ[name]
        except KeyError as exc:
            raise KeyError(f"Required secret environment variable '{name}' is not set") from exc


class InMemoryJobBroker:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[str, str, Mapping[str, Any]]] = asyncio.Queue()

    async def enqueue(self, tenant_id: str, payload: Mapping[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        await self.queue.put((job_id, tenant_id, payload))
        return job_id

    async def dequeue(self) -> tuple[str, str, Mapping[str, Any]]:
        return await self.queue.get()
