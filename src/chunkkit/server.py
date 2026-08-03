"""Optional multi-tenant FastAPI surface."""

import json
import os
import secrets
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .context import ContextPacker
from .errors import missing_extra
from .models import Chunk, ContextBundle, Document, ModelTarget, PipelineSpec, ScoredChunk
from .pipeline import ChunkingPipeline
from .plugins import PLUGIN_GROUPS, PluginManager


@dataclass(frozen=True, slots=True)
class AuthContext:
    tenant_id: str
    roles: frozenset[str]


class PreviewRequest(BaseModel):
    document: Document
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)


class PackRequest(BaseModel):
    chunks: list[ScoredChunk | Chunk]
    target: ModelTarget


class ResourceRequest(BaseModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class JobRequest(BaseModel):
    operation: Literal["chunk", "sync", "evaluate"]
    payload: dict[str, Any] = Field(default_factory=dict)


def _api_keys() -> dict[str, AuthContext]:
    configured = os.getenv("CHUNKKIT_API_KEYS")
    if configured:
        values = json.loads(configured)
        return {
            token: AuthContext(
                tenant_id=str(value["tenant_id"]),
                roles=frozenset(map(str, value.get("roles", ("runner",)))),
            )
            for token, value in values.items()
        }
    return {"dev": AuthContext("default", frozenset({"platform_admin", "runner", "viewer"}))}


def create_app():  # type: ignore[no-untyped-def]
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, status
    except ImportError as exc:  # pragma: no cover
        raise missing_extra("ChunkKit server", "server", "fastapi") from exc

    api_keys = _api_keys()
    app = FastAPI(title="ChunkKit", version="0.1.0", docs_url="/docs")
    resources: dict[str, dict[str, dict[str, Any]]] = {}
    jobs: dict[str, dict[str, Any]] = {}

    def authenticate(authorization: Annotated[str | None, Header()] = None) -> AuthContext:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token"
            )
        token = authorization.removeprefix("Bearer ")
        context = next(
            (value for key, value in api_keys.items() if secrets.compare_digest(key, token)), None
        )
        if context is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token"
            )
        return context

    Auth = Annotated[AuthContext, Depends(authenticate)]

    def authorize(auth: AuthContext, *required: str) -> None:
        if "platform_admin" in auth.roles or auth.roles.intersection(required):
            return
        raise HTTPException(status_code=403, detail="insufficient role")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/chunks:preview", response_model=list[Chunk])
    async def preview(request: PreviewRequest, auth: Auth) -> list[Chunk]:
        authorize(auth, "runner", "editor")
        if len(request.document.text) > 5_000_000:
            raise HTTPException(status_code=413, detail="demo preview limit is 5 MB")
        document = request.document.model_copy(update={"tenant_id": auth.tenant_id})
        return list(ChunkingPipeline.from_spec(request.pipeline).chunk(document))

    @app.post("/v1/contexts:pack", response_model=ContextBundle)
    async def pack(request: PackRequest, auth: Auth) -> ContextBundle:
        authorize(auth, "runner", "viewer")
        for item in request.chunks:
            chunk = item.chunk if isinstance(item, ScoredChunk) else item
            if chunk.tenant_id != auth.tenant_id:
                raise HTTPException(status_code=403, detail="cross-tenant chunk rejected")
        return ContextPacker(request.target).assemble(request.chunks)

    @app.get("/v1/plugins")
    async def plugins(auth: Auth) -> dict[str, list[str]]:
        authorize(auth, "viewer", "runner", "editor")
        manager = PluginManager()
        return {group: sorted(manager.discover(group)) for group in PLUGIN_GROUPS}

    @app.get("/v1/models")
    async def models(auth: Auth) -> dict[str, Any]:
        authorize(auth, "viewer", "runner", "editor")
        return {
            "profiles": [],
            "policy": "Explicit profiles are required; limits are never guessed.",
        }

    for resource_name in ("sources", "pipelines", "experiments", "artifacts"):

        async def list_resources(auth: Auth, resource: str = resource_name) -> list[dict[str, Any]]:
            authorize(auth, "viewer", "runner", "editor")
            return list(resources.get(auth.tenant_id, {}).get(resource, {}).values())

        async def create_resource(
            request: ResourceRequest, auth: Auth, resource: str = resource_name
        ) -> dict[str, Any]:
            authorize(auth, "editor")
            tenant = resources.setdefault(auth.tenant_id, {})
            collection = tenant.setdefault(resource, {})
            identifier = secrets.token_hex(12)
            value = {"id": identifier, "name": request.name, "config": request.config}
            collection[identifier] = value
            return value

        app.add_api_route(f"/v1/{resource_name}", list_resources, methods=["GET"])
        app.add_api_route(
            f"/v1/{resource_name}", create_resource, methods=["POST"], status_code=201
        )

    @app.post("/v1/jobs", status_code=202)
    async def create_job(request: JobRequest, auth: Auth) -> dict[str, Any]:
        authorize(auth, "runner")
        identifier = secrets.token_hex(12)
        jobs[identifier] = {
            "id": identifier,
            "tenant_id": auth.tenant_id,
            "status": "queued",
            "operation": request.operation,
            "payload": request.payload,
        }
        return {key: value for key, value in jobs[identifier].items() if key != "tenant_id"}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, auth: Auth) -> dict[str, Any]:
        authorize(auth, "viewer", "runner")
        value = jobs.get(job_id)
        if not value or value["tenant_id"] != auth.tenant_id:
            raise HTTPException(status_code=404, detail="job not found")
        return {key: item for key, item in value.items() if key != "tenant_id"}

    return app
