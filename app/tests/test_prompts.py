import os
import pytest

from app.prompts import Prompt


class TestPromptFormat:
    def test_replaces_single_key(self):
        p = Prompt("Hello {NAME}!", keys=["NAME"])
        assert p.format(NAME="world") == "Hello world!"

    def test_replaces_multiple_keys(self):
        p = Prompt("{A} and {B}", keys=["A", "B"])
        assert p.format(A="foo", B="bar") == "foo and bar"

    def test_converts_non_string_values(self):
        p = Prompt("size={SIZE}", keys=["SIZE"])
        assert p.format(SIZE=42) == "size=42"


class TestPromptOverhead:
    def test_overhead_excludes_placeholder_chars(self):
        p = Prompt("Hello {TEXT} world", keys=["TEXT"])
        # len("Hello {TEXT} world") = 18, len("{TEXT}") = 6 → overhead = 12
        assert p.overhead == 12

    def test_overhead_multiple_keys(self):
        p = Prompt("{A}{B}", keys=["A", "B"])
        # len("{A}") = 3, len("{B}") = 3, total template = 6 → overhead = 0
        assert p.overhead == 0

    def test_overhead_no_keys(self):
        p = Prompt("static text", keys=[])
        assert p.overhead == len("static text")


class TestPromptFromFile:
    def test_loads_template_from_file(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("say {WORD}")
        p = Prompt.from_file(str(f), keys=["WORD"])
        assert p.format(WORD="hello") == "say hello"

    def test_version_is_set(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("{X}")
        p = Prompt.from_file(str(f), keys=["X"], version=3)
        assert p.version == 3

    def test_version_defaults_to_none(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("{X}")
        p = Prompt.from_file(str(f), keys=["X"])
        assert p.version is None


class TestPromptLoad:
    def test_loads_versioned_file(self, tmp_path):
        prompt_dir = tmp_path / "resume"
        prompt_dir.mkdir()
        (prompt_dir / "v1.txt").write_text("resume {TEXT}")

        p = Prompt.load("resume", version=1, keys=["TEXT"], prompts_dir=str(tmp_path))

        assert p.format(TEXT="content") == "resume content"
        assert p.version == 1

    def test_loads_correct_version(self, tmp_path):
        prompt_dir = tmp_path / "resume"
        prompt_dir.mkdir()
        (prompt_dir / "v1.txt").write_text("version one {TEXT}")
        (prompt_dir / "v2.txt").write_text("version two {TEXT}")

        p1 = Prompt.load("resume", version=1, keys=["TEXT"], prompts_dir=str(tmp_path))
        p2 = Prompt.load("resume", version=2, keys=["TEXT"], prompts_dir=str(tmp_path))

        assert p1.format(TEXT="x") == "version one x"
        assert p2.format(TEXT="x") == "version two x"

    def test_raises_if_version_not_found(self, tmp_path):
        (tmp_path / "resume").mkdir()
        with pytest.raises(FileNotFoundError):
            Prompt.load("resume", version=99, keys=["TEXT"], prompts_dir=str(tmp_path))

    def test_version_attribute_is_stored(self, tmp_path):
        prompt_dir = tmp_path / "summarize"
        prompt_dir.mkdir()
        (prompt_dir / "v2.txt").write_text("{ABSTRACT_CONTENTS}")

        p = Prompt.load("summarize", version=2, keys=["ABSTRACT_CONTENTS"], prompts_dir=str(tmp_path))
        assert p.version == 2


class TestAvailableVersions:
    def test_returns_sorted_versions(self, tmp_path):
        d = tmp_path / "resume"
        d.mkdir()
        (d / "v3.txt").write_text("three")
        (d / "v1.txt").write_text("one")
        (d / "v2.txt").write_text("two")
        assert Prompt.available_versions("resume", prompts_dir=str(tmp_path)) == [1, 2, 3]

    def test_returns_empty_list_for_missing_dir(self, tmp_path):
        assert Prompt.available_versions("nonexistent", prompts_dir=str(tmp_path)) == []

    def test_ignores_non_versioned_files(self, tmp_path):
        d = tmp_path / "resume"
        d.mkdir()
        (d / "v1.txt").write_text("one")
        (d / "notes.txt").write_text("ignore me")
        (d / "v2.md").write_text("also ignored")
        assert Prompt.available_versions("resume", prompts_dir=str(tmp_path)) == [1]

    def test_single_version(self, tmp_path):
        d = tmp_path / "keywords"
        d.mkdir()
        (d / "v1.txt").write_text("one")
        assert Prompt.available_versions("keywords", prompts_dir=str(tmp_path)) == [1]
