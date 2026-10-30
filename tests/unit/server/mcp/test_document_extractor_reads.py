"""MCP text readers use extracted content without treating it as source code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from repowise.core.ingestion.document_extractors import document_extractors
from repowise.core.registry import plugins
from repowise.server.mcp_server._helpers import read_repo_file_text
from repowise.server.mcp_server.tool_answer.source import _read_repo_text
from repowise.server.mcp_server.tool_context.skeleton import _resolve_skeleton


@dataclass
class FakeExtractor:
    extensions: frozenset[str] = frozenset({".xyz"})
    version: str = "1"

    def extract(self, path: Path, data: bytes) -> str:
        return "# Extracted document\n\nReadable content.\n"


@pytest.fixture
def extractor_registry(monkeypatch):
    monkeypatch.setattr(document_extractors, "_impls", {})
    monkeypatch.setattr(document_extractors, "_chosen", None)
    monkeypatch.setattr(plugins, "load_plugins", lambda: [])
    document_extractors.register("mcp-test-extractor", FakeExtractor())


def test_generic_mcp_text_reader_returns_extracted_content(
    tmp_path, extractor_registry
):
    document = tmp_path / "spec.xyz"
    document.write_bytes(b"\x00binary source")

    assert read_repo_file_text(tmp_path, "spec.xyz") == (
        "# Extracted document\n\nReadable content.\n"
    )


def test_code_source_reader_skips_extracted_documents(tmp_path, extractor_registry):
    document = tmp_path / "spec.xyz"
    document.write_bytes(b"\x00binary source")

    assert _read_repo_text(tmp_path, "spec.xyz") is None


@pytest.mark.asyncio
async def test_skeleton_reader_skips_extracted_documents(tmp_path, extractor_registry):
    document = tmp_path / "spec.xyz"
    document.write_bytes(b"\x00binary source")
    result: dict = {}

    class EmptyResult:
        def scalars(self):
            return self

        def all(self):
            return []

    class EmptySession:
        async def execute(self, _query):
            return EmptyResult()

    await _resolve_skeleton(
        session=EmptySession(),
        repository=SimpleNamespace(id=1),
        target="spec.xyz",
        target_type="file",
        result_data=result,
        repo_root=tmp_path,
    )

    assert result["skeleton"]["error"] == (
        "Source skeletons are not available for extracted documents."
    )
