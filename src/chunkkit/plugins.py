"""Plugin discovery and registration."""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal

from pydantic import Field

from .errors import PluginError
from .models import FrozenModel

PLUGIN_GROUPS = (
    "chunkkit.chunkers",
    "chunkkit.connectors",
    "chunkkit.parsers",
    "chunkkit.tokenizers",
    "chunkkit.models",
    "chunkkit.vectorstores",
    "chunkkit.evaluators",
    "chunkkit.storage",
)


class PluginManifest(FrozenModel):
    name: str
    version: str
    api_version: str = "1"
    capabilities: tuple[str, ...] = ()
    config_schema: dict[str, Any] = Field(default_factory=dict)
    network_access: bool = False
    data_behavior: Literal["local", "remote", "mixed"] = "local"


class PluginManager:
    def __init__(self, *, allowlist: set[str] | None = None) -> None:
        self.allowlist = allowlist

    def discover(self, group: str) -> dict[str, EntryPoint]:
        if group not in PLUGIN_GROUPS:
            raise PluginError(f"Unknown plugin group '{group}'")
        discovered = entry_points(group=group)
        return {
            point.name: point
            for point in discovered
            if self.allowlist is None or f"{group}:{point.name}" in self.allowlist
        }

    def load(self, group: str, name: str) -> Any:
        point = self.discover(group).get(name)
        if point is None:
            allowed = " or it is not server-allowlisted" if self.allowlist is not None else ""
            raise PluginError(f"Plugin '{group}:{name}' is not installed{allowed}")
        try:
            return point.load()
        except Exception as exc:
            raise PluginError(f"Failed to load plugin '{group}:{name}': {exc}") from exc
