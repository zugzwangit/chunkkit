"""Versioned model profile registry without silent network lookups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .errors import ConfigurationError
from .models import ModelProfile, ModelTarget


class ModelProfileRegistry:
    def __init__(self, profiles: list[ModelProfile] | None = None) -> None:
        self._profiles: dict[tuple[str, str], ModelProfile] = {}
        for profile in profiles or []:
            self.register(profile)

    def register(self, profile: ModelProfile) -> None:
        self._profiles[(profile.provider, profile.model)] = profile

    def resolve(self, provider: str, model: str) -> ModelProfile:
        try:
            return self._profiles[(provider, model)]
        except KeyError as exc:
            raise ConfigurationError(
                f"No model profile for '{provider}/{model}'. Supply a versioned profile or an "
                "explicit ModelTarget; ChunkKit will not guess context limits."
            ) from exc

    def target(
        self,
        provider: str,
        model: str,
        *,
        role: Literal["embedding", "reranking", "generation", "generic"] = "generic",
        reserved_tokens: int = 1024,
        safety_margin_tokens: int = 64,
    ) -> ModelTarget:
        profile = self.resolve(provider, model)
        return ModelTarget(
            tokenizer=profile.tokenizer,
            model=profile.model,
            role=role,
            max_input_tokens=profile.max_input_tokens,
            reserved_tokens=reserved_tokens,
            safety_margin_tokens=safety_margin_tokens,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelProfileRegistry:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ConfigurationError("model profile files must contain a JSON array")
        return cls([ModelProfile.model_validate(value) for value in values])

    def list(self) -> tuple[ModelProfile, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))
