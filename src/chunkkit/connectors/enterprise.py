"""Configurable HTTP connectors for common enterprise APIs.

These connectors intentionally normalize records through a small common reader.
Deployments can subclass :class:`RestRecordConnector` when custom fields, ACLs, or
vendor-specific content expansion are required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from hashlib import sha256
from typing import Any

from ..errors import missing_extra
from ..models import (
    Acl,
    AclResolution,
    ChangeEvent,
    Checkpoint,
    Document,
    RawArtifact,
    UpsertArtifact,
    Visibility,
)

AclMapper = Callable[[Mapping[str, Any]], Acl]


def _plain(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (_plain(item) for item in value)))
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", ""))
        if "plain_text" in value:
            return str(value["plain_text"])
        if "value" in value and isinstance(value["value"], str):
            return value["value"]
        return "\n".join(filter(None, (_plain(item) for item in value.values())))
    return str(value)


def _coarse_acl(source: str) -> Acl:
    return Acl(
        visibility=Visibility.RESTRICTED,
        allow=(),
        resolution=AclResolution.INCOMPLETE,
        source=source,
    )


class RestRecordConnector:
    name = "rest"
    extra = "atlassian"
    records_key = "results"
    id_field = "id"
    title_fields: Sequence[str] = ("title", "name", "key")
    content_fields: Sequence[str] = ("body", "description", "content", "text", "fields")

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        endpoint: str,
        tenant_id: str = "default",
        headers: Mapping[str, str] | None = None,
        acl_mapper: AclMapper | None = None,
        page_size: int = 100,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.endpoint = endpoint
        self.tenant_id = tenant_id
        self.headers = dict(headers or {})
        self.acl_mapper = acl_mapper
        self.page_size = page_size

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        params = f"limit={self.page_size}"
        if cursor:
            params += f"&cursor={cursor}"
        return "GET", f"{self.base_url}{self.endpoint}?{params}", None

    def records(self, payload: Any) -> Sequence[Mapping[str, Any]]:
        if not isinstance(payload, Mapping):
            return ()
        value = payload.get(self.records_key, ())
        return (
            tuple(item for item in value if isinstance(item, dict))
            if isinstance(value, list)
            else ()
        )

    def next_cursor(self, payload: Mapping[str, Any]) -> str | None:
        links = payload.get("_links")
        if isinstance(links, dict) and links.get("next"):
            return str(links["next"])
        for key in ("next_cursor", "after_cursor", "nextPageToken"):
            if payload.get(key):
                return str(payload[key])
        return None

    def document(self, record: Mapping[str, Any]) -> Document:
        record_id = str(record.get(self.id_field, sha256(repr(record).encode()).hexdigest()))
        title = next(
            (_plain(record.get(field)) for field in self.title_fields if record.get(field)), None
        )
        parts = [title] if title else []
        parts.extend(
            _plain(record.get(field)) for field in self.content_fields if record.get(field)
        )
        text = "\n\n".join(part for part in parts if part).strip()
        acl = self.acl_mapper(record) if self.acl_mapper else _coarse_acl(self.name)
        return Document(
            id=f"{self.name}:{record_id}",
            text=text or json.dumps(record, ensure_ascii=False, sort_keys=True),
            source_uri=f"{self.base_url}{self.endpoint}/{record_id}",
            revision=str(
                record.get("version") or record.get("updated") or record.get("updated_at") or ""
            ),
            mime_type=f"application/vnd.chunkkit.{self.name}+json",
            tenant_id=self.tenant_id,
            title=title,
            metadata={"connector": self.name, "record_id": record_id},
            acl=acl,
        )

    async def sync(self, cursor: str | None = None) -> AsyncIterator[ChangeEvent]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise missing_extra(f"{self.name} connector", self.extra, "httpx") from exc
        current = cursor
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            while True:
                method, url, body = self.request(current)
                headers = {"Authorization": f"Bearer {self.token}", **self.headers}
                response = await client.request(method, url, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
                for record in self.records(payload):
                    document = self.document(record)
                    raw = json.dumps(record, ensure_ascii=False, sort_keys=True).encode()
                    artifact = RawArtifact(
                        source_uri=document.source_uri,
                        revision=document.revision,
                        mime_type=document.mime_type,
                        checksum=sha256(raw).hexdigest(),
                        metadata=document.metadata,
                    )
                    yield UpsertArtifact(artifact=artifact, document=document, cursor=current)
                next_value = self.next_cursor(payload) if isinstance(payload, Mapping) else None
                if not next_value or next_value == current:
                    break
                current = next_value
        yield Checkpoint(cursor=current or "complete")


class ConfluenceConnector(RestRecordConnector):
    name = "confluence"
    extra = "atlassian"

    def __init__(self, base_url: str, token: str, **kwargs: Any) -> None:
        super().__init__(base_url, token, endpoint="/wiki/api/v2/pages", **kwargs)

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        url = cursor if cursor and cursor.startswith("http") else f"{self.base_url}{self.endpoint}"
        separator = "&" if "?" in url else "?"
        return "GET", f"{url}{separator}limit={self.page_size}&body-format=storage", None


class JiraConnector(RestRecordConnector):
    name = "jira"
    extra = "atlassian"
    records_key = "issues"
    id_field = "key"
    title_fields = ("key",)
    content_fields = ("fields",)

    def __init__(
        self, base_url: str, token: str, *, jql: str = "ORDER BY updated", **kwargs: Any
    ) -> None:
        super().__init__(base_url, token, endpoint="/rest/api/3/search/jql", **kwargs)
        self.jql = jql

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        return (
            "POST",
            f"{self.base_url}{self.endpoint}",
            {
                "jql": self.jql,
                "maxResults": self.page_size,
                "nextPageToken": cursor,
                "fields": [
                    "summary",
                    "description",
                    "comment",
                    "attachment",
                    "updated",
                    "security",
                ],
            },
        )

    def next_cursor(self, payload: Mapping[str, Any]) -> str | None:
        value = payload.get("nextPageToken")
        return str(value) if value else None


class NotionConnector(RestRecordConnector):
    name = "notion"
    extra = "notion"
    records_key = "results"

    def __init__(self, base_url: str, token: str, **kwargs: Any) -> None:
        headers = {"Notion-Version": "2025-09-03"}
        super().__init__(base_url, token, endpoint="/v1/search", headers=headers, **kwargs)

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        return (
            "POST",
            f"{self.base_url}{self.endpoint}",
            {
                "page_size": self.page_size,
                "start_cursor": cursor,
            },
        )


class GoogleDriveConnector(RestRecordConnector):
    name = "google_drive"
    extra = "google"
    records_key = "files"

    def __init__(self, base_url: str, token: str, **kwargs: Any) -> None:
        super().__init__(base_url, token, endpoint="/drive/v3/files", **kwargs)

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        page = f"&pageToken={cursor}" if cursor else ""
        fields = "nextPageToken,files(id,name,mimeType,modifiedTime,description,permissions)"
        return (
            "GET",
            (f"{self.base_url}{self.endpoint}?pageSize={self.page_size}&fields={fields}{page}"),
            None,
        )


class MicrosoftGraphConnector(RestRecordConnector):
    name = "microsoft_graph"
    extra = "microsoft"
    records_key = "value"

    def __init__(self, base_url: str, token: str, *, resource: str, **kwargs: Any) -> None:
        super().__init__(base_url, token, endpoint=resource, **kwargs)

    def next_cursor(self, payload: Mapping[str, Any]) -> str | None:
        value = payload.get("@odata.nextLink")
        return str(value) if value else None

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        if cursor and cursor.startswith("http"):
            return "GET", cursor, None
        return super().request(cursor)


class SlackConnector(RestRecordConnector):
    name = "slack"
    extra = "slack"
    records_key = "messages"

    def __init__(self, base_url: str, token: str, *, channel: str, **kwargs: Any) -> None:
        super().__init__(base_url, token, endpoint="/api/conversations.history", **kwargs)
        self.channel = channel

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        suffix = f"&cursor={cursor}" if cursor else ""
        return (
            "GET",
            (
                f"{self.base_url}{self.endpoint}?channel={self.channel}&limit={self.page_size}{suffix}"
            ),
            None,
        )

    def next_cursor(self, payload: Mapping[str, Any]) -> str | None:
        metadata = payload.get("response_metadata")
        return (
            str(metadata.get("next_cursor"))
            if isinstance(metadata, dict) and metadata.get("next_cursor")
            else None
        )


class GitHubConnector(RestRecordConnector):
    name = "github"
    extra = "github"
    records_key = "items"

    def __init__(self, base_url: str, token: str, *, repository: str, **kwargs: Any) -> None:
        super().__init__(base_url, token, endpoint=f"/repos/{repository}/issues", **kwargs)

    def records(self, payload: Any) -> Sequence[Mapping[str, Any]]:
        if isinstance(payload, list):
            return tuple(item for item in payload if isinstance(item, dict))
        return super().records(payload)


class ServiceNowConnector(RestRecordConnector):
    name = "servicenow"
    extra = "servicenow"
    records_key = "result"
    id_field = "sys_id"
    title_fields = ("number", "short_description")
    content_fields = ("short_description", "description", "comments_and_work_notes")

    def __init__(
        self, base_url: str, token: str, *, table: str = "incident", **kwargs: Any
    ) -> None:
        super().__init__(base_url, token, endpoint=f"/api/now/table/{table}", **kwargs)


class ZendeskConnector(RestRecordConnector):
    name = "zendesk"
    extra = "zendesk"
    records_key = "tickets"
    title_fields = ("subject",)
    content_fields = ("description", "comment_events")

    def __init__(self, base_url: str, token: str, **kwargs: Any) -> None:
        super().__init__(
            base_url, token, endpoint="/api/v2/incremental/tickets/cursor.json", **kwargs
        )

    def request(self, cursor: str | None) -> tuple[str, str, dict[str, Any] | None]:
        value = f"cursor={cursor}" if cursor else "start_time=0"
        return "GET", f"{self.base_url}{self.endpoint}?{value}&include=comment_events", None

    def next_cursor(self, payload: Mapping[str, Any]) -> str | None:
        value = payload.get("after_cursor")
        return str(value) if value and not payload.get("end_of_stream") else None
