"""Built-in lightweight parsers and optional document parsers."""

from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from .errors import ConfigurationError, missing_extra
from .models import Document, Element, ElementKind, RawArtifact, SourceSpan
from .protocols import Parser


class _TextHTMLParser(HTMLParser):
    block_tags: ClassVar[set[str]] = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.block_tags and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", value)
        return value.strip()


class TextParser:
    name = "text"

    async def parse(self, artifact: RawArtifact, content: bytes) -> Document:
        text = content.decode("utf-8", errors="replace")
        return Document(
            text=text,
            source_uri=artifact.source_uri,
            revision=artifact.revision,
            mime_type=artifact.mime_type,
            metadata=artifact.metadata,
        )


class HTMLTextParser:
    name = "html"

    async def parse(self, artifact: RawArtifact, content: bytes) -> Document:
        parser = _TextHTMLParser()
        parser.feed(content.decode("utf-8", errors="replace"))
        return Document(
            text=parser.text(),
            source_uri=artifact.source_uri,
            revision=artifact.revision,
            mime_type=artifact.mime_type,
            metadata=artifact.metadata,
        )


class JSONParser:
    name = "json"

    async def parse(self, artifact: RawArtifact, content: bytes) -> Document:
        value = json.loads(content)
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        elements: list[Element] = []
        if isinstance(value, dict):
            for key, item in value.items():
                rendered = json.dumps(item, ensure_ascii=False, sort_keys=True)
                start = text.find(f'"{key}"')
                elements.append(
                    Element(
                        kind=ElementKind.OTHER,
                        text=rendered,
                        span=SourceSpan(pointer=f"/{key}", start=max(0, start), end=len(text)),
                        metadata={"key": key},
                    )
                )
        return Document(
            text=text,
            source_uri=artifact.source_uri,
            revision=artifact.revision,
            mime_type=artifact.mime_type,
            elements=tuple(elements),
            metadata=artifact.metadata,
        )


class DelimitedParser:
    def __init__(self, delimiter: str = ",") -> None:
        self.delimiter = delimiter
        self.name = "csv" if delimiter == "," else "tsv"

    async def parse(self, artifact: RawArtifact, content: bytes) -> Document:
        source = content.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(source), delimiter=self.delimiter))
        rendered = "\n".join(" | ".join(cell for cell in row) for row in rows)
        element = Element(
            kind=ElementKind.TABLE,
            text=rendered,
            span=SourceSpan(start=0, end=len(rendered)),
            metadata={"rows": len(rows), "columns": max((len(row) for row in rows), default=0)},
        )
        return Document(
            text=rendered,
            source_uri=artifact.source_uri,
            revision=artifact.revision,
            mime_type=artifact.mime_type,
            elements=(element,),
            metadata=artifact.metadata,
        )


class PDFParser:
    name = "pdf"

    async def parse(self, artifact: RawArtifact, content: bytes) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise missing_extra("PDF parsing", "documents", "pypdf") from exc
        reader = PdfReader(io.BytesIO(content))
        parts: list[str] = []
        elements: list[Element] = []
        cursor = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if parts:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(page_text)
            cursor += len(page_text)
            elements.append(
                Element(
                    kind=ElementKind.PARAGRAPH,
                    text=page_text,
                    span=SourceSpan(start=start, end=cursor, page=page_number),
                )
            )
        return Document(
            text="".join(parts),
            source_uri=artifact.source_uri,
            revision=artifact.revision,
            mime_type=artifact.mime_type,
            elements=tuple(elements),
            metadata=artifact.metadata,
        )


PARSERS: dict[str, Parser] = {
    "text/plain": TextParser(),
    "text/markdown": TextParser(),
    "text/html": HTMLTextParser(),
    "application/xhtml+xml": HTMLTextParser(),
    "application/json": JSONParser(),
    "text/csv": DelimitedParser(","),
    "text/tab-separated-values": DelimitedParser("\t"),
    "application/pdf": PDFParser(),
}

EXTENSION_MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".mdx": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".pdf": "application/pdf",
    ".py": "text/x-python",
    ".js": "text/x-javascript",
    ".ts": "text/x-typescript",
    ".java": "text/x-java",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
}


def parser_for(mime_type: str) -> Parser:
    if mime_type.startswith("text/x-"):
        return TextParser()
    try:
        return PARSERS[mime_type.partition(";")[0].strip().lower()]
    except KeyError as exc:
        raise ConfigurationError(
            f"No built-in parser for '{mime_type}'. Install a document extra or register a "
            "'chunkkit.parsers' plugin."
        ) from exc


async def parse_path(path: str | Path, *, tenant_id: str = "default") -> Document:
    target = Path(path).resolve()
    mime_type = EXTENSION_MIME_TYPES.get(target.suffix.lower(), "text/plain")
    content = target.read_bytes()
    artifact = RawArtifact(
        source_uri=target.as_uri(),
        revision=str(target.stat().st_mtime_ns),
        mime_type=mime_type,
        checksum=__import__("hashlib").sha256(content).hexdigest(),
        metadata={"tenant_id": tenant_id, "filename": target.name},
    )
    document: Document = await parser_for(mime_type).parse(artifact, content)
    return document.model_copy(update={"tenant_id": tenant_id})
