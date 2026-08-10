"""Build-context staging: determinism across processes and temp-dir hygiene."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from containerspec import BuildError, ImageSpec, MissingToolError

_CONTEXT_PREFIX = "containerspec-context-"

_STAGE_SCRIPT = """
import shutil, sys
from containerspec import ImageSpec

src = sys.argv[1]
spec = ImageSpec.from_registry("python:3.12", pin_digest=False).copy(src, "/app/app.py")
context_dir, dockerfile = spec._prepare_build_context(spec.to_dockerfile())
try:
    sys.stdout.write(dockerfile)
finally:
    shutil.rmtree(context_dir, ignore_errors=True)
"""


# Exercises the naming helper directly, so it can cover source strings that the
# COPY path allowlist rejects before they ever reach a Dockerfile.
_NAME_SCRIPT = """
import sys
from containerspec.spec import _staged_name
from pathlib import PurePosixPath

for src in sys.argv[1:]:
    sys.stdout.write(_staged_name(src, basename=PurePosixPath(src).name) + "\\n")
"""

_TRICKY_SOURCES = (
    "./src",
    "src",
    "/abs/path/pkg",
    "/abs/path/pkg/",
    "café/mödule.py",
    "日本語/ファイル",
    "a b/c\td",
    "",
)


def _run_with_seed(script: str, args: list[str], seed: str) -> str:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
    )
    return result.stdout


def _stage_with_seed(src: str, seed: str) -> str:
    return _run_with_seed(_STAGE_SCRIPT, [src], seed)


def _leaked_context_dirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(f"{_CONTEXT_PREFIX}*"))


def _copy_targets(dockerfile: str) -> dict[str, str]:
    """Map each COPY destination to the source it reads from."""
    targets: dict[str, str] = {}
    for line in dockerfile.splitlines():
        if line.startswith("COPY "):
            src, dest = line.removeprefix("COPY ").rsplit(" ", 1)
            targets[dest] = src
    return targets


def _staged_entries(context_dir: str) -> list[str]:
    return sorted(p.name for p in Path(context_dir).iterdir() if p.name != "Dockerfile")


class TestStagedContextDeterminism:
    def test_staged_names_stable_across_python_hash_seeds(self, tmp_path: Path) -> None:
        """Staged filenames must not depend on PYTHONHASHSEED.

        Built-in ``hash()`` is salted per process for str, so deriving staged
        names from it makes the emitted Dockerfile — and therefore the image the
        backend produces — differ between otherwise identical runs.
        """
        src = tmp_path / "app.py"
        src.write_text("print('hi')\n")

        outputs = {_stage_with_seed(str(src), seed) for seed in ("1", "2", "12345")}

        assert len(outputs) == 1, f"staged Dockerfile varies across hash seeds: {outputs}"
        assert "COPY ctx_" in outputs.pop()

    def test_staged_name_stable_for_awkward_sources(self) -> None:
        """Non-ASCII, whitespace, empty and trailing-separator sources hash stably.

        ``validate_path`` rejects most of these before they reach a Dockerfile,
        so they are exercised against the naming helper directly — the helper
        must not acquire a hidden dependence on locale or hash seed.
        """
        args = list(_TRICKY_SOURCES)

        outputs = {_run_with_seed(_NAME_SCRIPT, args, seed) for seed in ("1", "2", "12345")}

        assert len(outputs) == 1, f"staged names vary across hash seeds: {outputs}"
        names = outputs.pop().splitlines()
        assert len(names) == len(args)
        # Distinct source strings must not collide, including "src" vs "./src"
        # and a path with and without its trailing separator.
        assert len(set(names)) == len(args)

    def test_staged_name_matches_sha256_of_source(self) -> None:
        """Pin the naming scheme so a future refactor cannot silently change it."""
        from containerspec.spec import _staged_name

        expected = hashlib.sha256(b"./src").hexdigest()[:8]

        assert _staged_name("./src", basename="src") == f"ctx_{expected}_src"


class TestStagedContextCleanup:
    @pytest.mark.asyncio
    async def test_build_removes_staged_context(self, tmp_path: Path) -> None:
        src = tmp_path / "app.py"
        src.write_text("print('hi')\n")
        client = MagicMock()
        client.images.get.side_effect = Exception("not found")
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).copy(
            str(src), "/app/app.py"
        )
        before = _leaked_context_dirs()

        await spec.build("warden/test", client=client, backend=AsyncMock())

        assert _leaked_context_dirs() == before

    @pytest.mark.asyncio
    async def test_build_removes_staged_context_on_failure(self, tmp_path: Path) -> None:
        src = tmp_path / "app.py"
        src.write_text("print('hi')\n")
        client = MagicMock()
        client.images.get.side_effect = Exception("not found")
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).copy(
            str(src), "/app/app.py"
        )
        backend = AsyncMock()
        backend.solve_and_export.side_effect = RuntimeError("boom")
        before = _leaked_context_dirs()

        with pytest.raises(BuildError):
            await spec.build("warden/test", client=client, backend=backend)

        assert _leaked_context_dirs() == before

    @pytest.mark.asyncio
    async def test_build_removes_staged_context_on_missing_tool(self, tmp_path: Path) -> None:
        """MissingToolError/BuildError re-raise ahead of the generic handler.

        That is a separate exit from ``build()``, so it needs its own proof that
        the staged sources are still cleaned up.
        """
        src = tmp_path / "app.py"
        src.write_text("print('hi')\n")
        client = MagicMock()
        client.images.get.side_effect = Exception("not found")
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).copy(
            str(src), "/app/app.py"
        )
        backend = AsyncMock()
        backend.solve_and_export.side_effect = MissingToolError("no buildah")
        before = _leaked_context_dirs()

        with pytest.raises(MissingToolError):
            await spec.build("warden/test", client=client, backend=backend)

        assert _leaked_context_dirs() == before


class TestStagedContextDuplicateSources:
    def test_same_source_copied_twice_stages_once_and_rewrites_both(self, tmp_path: Path) -> None:
        """One source, two destinations: staged once, both COPY lines rewritten.

        Previously the second ``copytree`` hit the already-staged name and
        raised ``FileExistsError``, failing the build outright.
        """
        src = tmp_path / "pkg"
        src.mkdir()
        (src / "mod.py").write_text("x = 1\n")
        spec = (
            ImageSpec.from_registry("python:3.12", pin_digest=False)
            .copy(str(src), "/a")
            .copy(str(src), "/b")
        )

        context_dir, dockerfile = spec._prepare_build_context(spec.to_dockerfile())
        try:
            staged = _staged_entries(context_dir)
            assert len(staged) == 1
            # Both destinations must be served from the one staged copy, and it
            # must carry the real contents — not merely exist.
            targets = _copy_targets(dockerfile)
            assert targets["/a"] == staged[0]
            assert targets["/b"] == staged[0]
            # No COPY instruction may still point at the host path (the ``# copy(...)``
            # provenance comment legitimately keeps it).
            assert f"COPY {src}" not in dockerfile
            assert (Path(context_dir) / staged[0] / "mod.py").read_text() == "x = 1\n"
            # The context Dockerfile on disk must match what the backend is told.
            assert (Path(context_dir) / "Dockerfile").read_text() == dockerfile
        finally:
            shutil.rmtree(context_dir, ignore_errors=True)

    def test_distinct_sources_sharing_a_basename_stay_separate(self, tmp_path: Path) -> None:
        """Same basename, different parents: two staged copies, no clobbering."""
        first = tmp_path / "one" / "pkg"
        second = tmp_path / "two" / "pkg"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / "mod.py").write_text("first\n")
        (second / "mod.py").write_text("second\n")
        spec = (
            ImageSpec.from_registry("python:3.12", pin_digest=False)
            .copy(str(first), "/a")
            .copy(str(second), "/b")
        )

        context_dir, dockerfile = spec._prepare_build_context(spec.to_dockerfile())
        try:
            targets = _copy_targets(dockerfile)
            assert targets["/a"] != targets["/b"]
            assert _staged_entries(context_dir) == sorted({targets["/a"], targets["/b"]})
            assert (Path(context_dir) / targets["/a"] / "mod.py").read_text() == "first\n"
            assert (Path(context_dir) / targets["/b"] / "mod.py").read_text() == "second\n"
        finally:
            shutil.rmtree(context_dir, ignore_errors=True)

    def test_trailing_separator_source_is_staged_and_rewritten(self, tmp_path: Path) -> None:
        """``copy("pkg/", ...)`` must stage under a real basename, not an empty one."""
        src = tmp_path / "pkg"
        src.mkdir()
        (src / "mod.py").write_text("x = 1\n")
        with_sep = f"{src}/"
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).copy(with_sep, "/a")

        context_dir, dockerfile = spec._prepare_build_context(spec.to_dockerfile())
        try:
            staged = _staged_entries(context_dir)
            assert staged == [_copy_targets(dockerfile)["/a"]]
            assert staged[0].endswith("_pkg")
            assert (Path(context_dir) / staged[0] / "mod.py").read_text() == "x = 1\n"
        finally:
            shutil.rmtree(context_dir, ignore_errors=True)
