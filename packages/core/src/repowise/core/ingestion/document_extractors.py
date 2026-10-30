"""Plugin seam for extracting Markdown from otherwise unsupported files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from repowise.core.registry.plugins import SeamRegistry

from .models import FileInfo


class DocumentExtractor(Protocol):
    """Convert a claimed source file into Markdown for the existing parser."""

    extensions: frozenset[str]
    version: str

    def extract(self, path: Path, data: bytes) -> str: ...


document_extractors: SeamRegistry[DocumentExtractor] = SeamRegistry("document-extractors")


class DocumentExtractionError(RuntimeError):
    """An extractor failed for a source path."""


def _extension_claims() -> dict[str, tuple[str, DocumentExtractor]]:
    claims: dict[str, tuple[str, DocumentExtractor]] = {}
    for name, extractor in document_extractors.all():
        for extension in extractor.extensions:
            if not isinstance(extension, str):
                raise ValueError(f"document extractor {name!r} has invalid extension {extension!r}")
            if (
                not extension.startswith(".")
                or extension != extension.lower()
                or len(extension) < 2
                or "/" in extension
                or "\\" in extension
            ):
                raise ValueError(f"document extractor {name!r} has invalid extension {extension!r}")
            claims[extension] = (name, extractor)
    return claims


def document_extractor_claims() -> dict[str, str]:
    """Map each unsupported extension to the implementation that will handle it."""
    from .traverser import _language_from_name_or_ext

    return {
        extension: name
        for extension, (name, _impl) in _extension_claims().items()
        if _language_from_name_or_ext(Path(f"source{extension}")) is None
    }


def document_extractor_for_path(
    path: Path | str, *, language: str | None = None
) -> tuple[str, DocumentExtractor] | None:
    """Return the later-registered extractor for an unsupported extension."""
    path = Path(path)
    from .traverser import _language_from_name_or_ext

    if language not in (None, "unknown"):
        return None
    if _language_from_name_or_ext(path) is not None:
        return None
    return _extension_claims().get(path.suffix.lower())


def is_extracted_document(path: Path | str) -> bool:
    """Whether *path* is routed through a document extractor."""
    return document_extractor_for_path(path) is not None


def source_fingerprint(file_info: FileInfo, source: bytes) -> str:
    """Hash original bytes and extractor identity/version without extracting."""
    name = file_info.extractor_name
    version = file_info.extractor_version
    if name is None or version is None:
        return hashlib.sha256(source).hexdigest()
    digest = hashlib.sha256()
    digest.update(b"repowise-document-extractor\0")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(source)
    return digest.hexdigest()


def extract_document_source(file_info: FileInfo, source: bytes) -> bytes:
    """Return extracted Markdown bytes, raising a named error on failure."""
    name = file_info.extractor_name
    if name is None:
        return source
    return _extract(
        name,
        dict(document_extractors.all())[name],
        Path(file_info.abs_path),
        file_info.path,
        source,
    )


def extract_document(path: Path | str, source: bytes) -> bytes:
    """Extract a claimed unsupported file for a text-serving caller."""
    from .traverser import _detect_language

    path = Path(path)
    claim = document_extractor_for_path(path, language=_detect_language(path, source))
    if claim is None:
        return source
    name, extractor = claim
    path = Path(path)
    return _extract(name, extractor, path, str(path), source)


def _extract(
    name: str,
    extractor: DocumentExtractor,
    extractor_path: Path,
    display_path: str,
    source: bytes,
) -> bytes:
    try:
        markdown = extractor.extract(extractor_path, source)
        if not isinstance(markdown, str):
            raise TypeError("extract() must return Markdown as str")
        return markdown.encode("utf-8")
    except Exception as exc:
        raise DocumentExtractionError(
            f"document extractor {name!r} failed for {display_path}: {exc}"
        ) from exc
