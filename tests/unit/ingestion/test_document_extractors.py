"""Plugin routing and cache behaviour for unsupported document types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from repowise.cli.commands.plugins_cmd import plugins_command
from repowise.core.ingestion.change_detector import ChangeDetector
from repowise.core.ingestion.document_extractors import (
    DocumentExtractionError,
    document_extractors,
)
from repowise.core.ingestion.traverser import FileTraverser
from repowise.core.pipeline.phases import ingestion as ingestion_phase
from repowise.core.registry import plugins


@dataclass
class FakeExtractor:
    extensions: frozenset[str] = frozenset({".xyz"})
    version: str = "1"
    output: str = "# Extracted\n\nThe document content.\n"
    calls: int = 0
    failure: Exception | None = None

    def extract(self, path: Path, data: bytes) -> str:
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.output


@pytest.fixture
def extractor_registry(monkeypatch):
    monkeypatch.setattr(document_extractors, "_impls", {})
    monkeypatch.setattr(document_extractors, "_chosen", None)
    monkeypatch.setattr(plugins, "load_plugins", lambda: [])


def _register(name: str, extractor: FakeExtractor) -> None:
    document_extractors.register(name, extractor)


async def _ingest(repo: Path):
    return await ingestion_phase._run_ingestion(
        repo,
        exclude_patterns=None,
        skip_tests=False,
        skip_infra=False,
        progress=None,
    )


def test_traversal_routes_only_unsupported_claims(tmp_path, extractor_registry):
    first = FakeExtractor()
    later = FakeExtractor(output="# Later extractor\n")
    python_extractor = FakeExtractor(extensions=frozenset({".py"}))
    _register("first", first)
    _register("later", later)
    _register("python-override", python_extractor)
    (tmp_path / "spec.xyz").write_bytes(b"\x00binary source")
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "script.xyz").write_text("#!/usr/bin/env python3\nVALUE = 2\n", encoding="utf-8")

    files = {info.path: info for info in FileTraverser(tmp_path).traverse()}

    assert files["spec.xyz"].language == "markdown"
    assert files["spec.xyz"].extractor_name == "later"
    assert files["spec.xyz"].extractor_version == "1"
    assert files["main.py"].language == "python"
    assert files["main.py"].extractor_name is None
    assert files["script.xyz"].language == "python"
    assert files["script.xyz"].extractor_name is None
    assert first.calls == later.calls == python_extractor.calls == 0


@pytest.mark.asyncio
async def test_parse_cache_skips_extraction_until_source_or_version_changes(
    tmp_path, extractor_registry
):
    extractor = FakeExtractor()
    _register("test-extractor", extractor)
    path = tmp_path / "docs" / "spec.xyz"
    path.parent.mkdir()
    path.write_bytes(b"source-v1")

    first, _file_infos, _structure, source_map, *_ = await _ingest(tmp_path)
    assert extractor.calls == 1
    extracted = next(parsed for parsed in first if parsed.file_info.path == "docs/spec.xyz")
    assert extracted.file_info.extracted_source == extractor.output.encode()
    assert source_map["docs/spec.xyz"] == extractor.output.encode()
    assert not path.with_suffix(".md").exists()

    await _ingest(tmp_path)
    assert extractor.calls == 1

    path.write_bytes(b"source-v2")
    await _ingest(tmp_path)
    assert extractor.calls == 2

    extractor.version = "2"
    await _ingest(tmp_path)
    assert extractor.calls == 3


def test_extractor_failure_names_plugin_and_path(tmp_path, extractor_registry):
    extractor = FakeExtractor(failure=ValueError("conversion failed"))
    _register("failing-extractor", extractor)
    path = tmp_path / "spec.xyz"
    path.write_bytes(b"\x00binary source")
    file_info = next(iter(FileTraverser(tmp_path).traverse()))

    with pytest.raises(DocumentExtractionError, match=r"failing-extractor.*spec\.xyz"):
        ingestion_phase._split_cached(tmp_path, [(file_info, path.read_bytes())], None)


def test_exclusions_and_size_limit_run_before_extraction(tmp_path, extractor_registry):
    extractor = FakeExtractor()
    _register("test-extractor", extractor)
    path = tmp_path / "spec.xyz"
    path.write_bytes(b"\x00binary source")

    assert list(FileTraverser(tmp_path, extra_exclude_patterns=["*.xyz"]).traverse()) == []
    assert list(FileTraverser(tmp_path, max_file_size_kb=0).traverse()) == []
    assert extractor.calls == 0


def test_plugins_command_shows_effective_extension_owner(extractor_registry):
    _register("test-extractor", FakeExtractor())

    result = CliRunner().invoke(plugins_command, ["--format", "json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["document_extractors"] == {".xyz": "test-extractor"}


def test_change_detector_uses_extracted_markdown_and_original_path(tmp_path, extractor_registry):
    extractor = FakeExtractor()
    _register("test-extractor", extractor)
    path = tmp_path / "spec.xyz"
    path.write_bytes(b"\x00binary source")

    parsed = ChangeDetector(tmp_path).parse_bytes(path.read_bytes(), "spec.xyz")

    assert parsed is not None
    assert parsed.file_info.path == "spec.xyz"
    assert parsed.file_info.language == "markdown"
    assert parsed.content_hash != hashlib.sha256(extractor.output.encode()).hexdigest()
    assert extractor.calls == 1
