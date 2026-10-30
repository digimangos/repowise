"""MCP text readers use extracted content without treating it as source code."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from repowise.core.ingestion.document_extractors import document_extractors
from repowise.core.registry import plugins
from repowise.server.mcp_server._helpers import read_repo_file_text
from repowise.server.mcp_server.tool_answer.source import _read_repo_text
from repowise.server.mcp_server.tool_context.skeleton import _resolve_skeleton
from tests.unit.ingestion.document_extractor_spawn_plugin import SpawnExtractor


@pytest.fixture
def extractor_registry(monkeypatch):
    monkeypatch.setattr(document_extractors, "_impls", {})
    monkeypatch.setattr(document_extractors, "_chosen", None)
    monkeypatch.setattr(plugins, "load_plugins", lambda: [])
    monkeypatch.setenv(
        "REPOWISE_PLUGINS", "tests.unit.ingestion.document_extractor_spawn_plugin:register"
    )
    monkeypatch.setenv("REPOWISE_TEST_EXTRACTOR_CALLS", "")
    monkeypatch.setenv("REPOWISE_TEST_EXTRACTOR_FAIL", "0")
    document_extractors.register("spawn-test-extractor", SpawnExtractor())


def test_generic_mcp_text_reader_returns_extracted_content(tmp_path, extractor_registry):
    document = tmp_path / "spec.xyz"
    document.write_bytes(b"\x00binary source")

    assert read_repo_file_text(tmp_path, "spec.xyz") == (
        "# Extracted document\n\n\x00binary source\n"
    )


def test_code_source_reader_skips_extracted_documents(tmp_path, extractor_registry):
    document = tmp_path / "spec.xyz"
    document.write_bytes(b"\x00binary source")

    assert _read_repo_text(tmp_path, "spec.xyz") is None


def test_code_source_reader_keeps_a_shebang_script_on_the_code_path(tmp_path, extractor_registry):
    script = tmp_path / "script.xyz"
    script.write_text("#!/usr/bin/env python3\nVALUE = 1\n", encoding="utf-8")

    assert _read_repo_text(tmp_path, "script.xyz") == script.read_text(encoding="utf-8")


def test_extracted_mcp_reads_enforce_input_budget_before_plugin_call(
    tmp_path, extractor_registry, monkeypatch
):
    from repowise.core.ingestion.traverser import _DEFAULT_MAX_FILE_SIZE_BYTES

    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("REPOWISE_TEST_EXTRACTOR_CALLS", str(calls))
    document = tmp_path / "spec.xyz"
    document.write_bytes(b"x" * (_DEFAULT_MAX_FILE_SIZE_BYTES + 1))

    assert read_repo_file_text(tmp_path, "spec.xyz") is None
    assert not calls.exists()


def test_literal_scan_budgets_include_skipped_documents(tmp_path, extractor_registry, monkeypatch):
    from repowise.server.mcp_server import _line_hits

    (tmp_path / "spec.xyz").write_bytes(b"document")
    monkeypatch.setattr(_line_hits, "MAX_SCAN_FILES", 0)
    rows, reasons = _line_hits.scan_literal(tmp_path, ["spec.xyz"], "document", False)
    assert rows == []
    assert reasons == ["over 0 files"]
    monkeypatch.setattr(_line_hits, "MAX_SCAN_FILES", 10)
    monkeypatch.setattr(_line_hits, "MAX_SCAN_BYTES", 0)
    rows, reasons = _line_hits.scan_literal(tmp_path, ["spec.xyz"], "document", False)
    assert rows == []
    assert "over 0 MB" in reasons[0]


def test_literal_scan_checks_time_before_document_skip(tmp_path, extractor_registry, monkeypatch):
    from repowise.server.mcp_server import _line_hits

    (tmp_path / "spec.xyz").write_bytes(b"document")
    clock = iter([0.0, _line_hits.MAX_SCAN_SECONDS + 1])
    monkeypatch.setattr(_line_hits.time, "monotonic", lambda: next(clock))
    rows, reasons = _line_hits.scan_literal(tmp_path, ["spec.xyz"], "document", False)
    assert rows == []
    assert "time budget" in reasons[0]


def test_literal_scan_keeps_supported_claimed_suffix_scripts(tmp_path, extractor_registry):
    from repowise.server.mcp_server._line_hits import scan_literal

    (tmp_path / "script.xyz").write_bytes(b"#!/usr/bin/env python3\nVALUE = 1\n")
    (tmp_path / "spec.xyz").write_bytes(b"VALUE = document\n")
    rows, reasons = scan_literal(tmp_path, ["spec.xyz", "script.xyz"], "VALUE", False)
    assert reasons == []
    assert [row["path"] for row in rows] == ["script.xyz"]


def test_source_guards_do_not_read_outside_root(tmp_path, extractor_registry, monkeypatch):
    from repowise.server.mcp_server import _edit_sites

    outside = tmp_path / "outside.xyz"
    outside.write_bytes(b"document")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "link.xyz").symlink_to(outside)
    assert _read_repo_text(root, "link.xyz") is None
    assert _edit_sites._live_definition(
        root, SimpleNamespace(file_path="../outside.xyz"), "name", "python"
    ) == (0, 0)


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
