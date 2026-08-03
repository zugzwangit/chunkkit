from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from chunkkit.server import create_app


def test_health_and_authentication(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "CHUNKKIT_API_KEYS",
        json.dumps({"alpha": {"tenant_id": "tenant-a", "roles": ["runner", "viewer"]}}),
    )
    client = TestClient(create_app())
    assert client.get("/health/ready").status_code == 200
    assert client.get("/v1/plugins").status_code == 401
    assert client.get("/v1/plugins", headers={"Authorization": "Bearer alpha"}).status_code == 200


def test_preview_overrides_untrusted_tenant(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "CHUNKKIT_API_KEYS",
        json.dumps({"alpha": {"tenant_id": "tenant-a", "roles": ["runner"]}}),
    )
    client = TestClient(create_app())
    response = client.post(
        "/v1/chunks:preview",
        headers={"Authorization": "Bearer alpha"},
        json={
            "document": {"text": "one two three", "tenant_id": "attacker"},
            "pipeline": {
                "chunk_size": 2,
                "overlap": 0,
                "target": {
                    "max_input_tokens": 16,
                    "reserved_tokens": 1,
                    "safety_margin_tokens": 1,
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    assert {item["tenant_id"] for item in response.json()} == {"tenant-a"}


def test_context_pack_rejects_cross_tenant(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "CHUNKKIT_API_KEYS",
        json.dumps({"alpha": {"tenant_id": "tenant-a", "roles": ["runner"]}}),
    )
    client = TestClient(create_app())
    preview = client.post(
        "/v1/chunks:preview",
        headers={"Authorization": "Bearer alpha"},
        json={"document": {"text": "one two"}},
    ).json()
    preview[0]["tenant_id"] = "tenant-b"
    response = client.post(
        "/v1/contexts:pack",
        headers={"Authorization": "Bearer alpha"},
        json={
            "chunks": [preview[0]],
            "target": {
                "max_input_tokens": 32,
                "reserved_tokens": 1,
                "safety_margin_tokens": 1,
            },
        },
    )
    assert response.status_code == 403


def test_rbac_rejects_viewer_write(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "CHUNKKIT_API_KEYS",
        json.dumps({"reader": {"tenant_id": "tenant-a", "roles": ["viewer"]}}),
    )
    client = TestClient(create_app())
    response = client.post(
        "/v1/sources",
        headers={"Authorization": "Bearer reader"},
        json={"name": "blocked"},
    )
    assert response.status_code == 403
