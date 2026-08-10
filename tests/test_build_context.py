"""Build-context staging: determinism across processes and temp-dir hygiene."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from containerspec import BuildError, ImageSpec

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


def _stage_with_seed(src: str, seed: str) -> str:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _STAGE_SCRIPT, src],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    )
    return result.stdout


def _leaked_context_dirs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(f"{_CONTEXT_PREFIX}*"))


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


class TestStagedContextDuplicateSources:
    def test_same_source_copied_twice_stages_once(self, tmp_path: Path) -> None:
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
            staged = [p.name for p in Path(context_dir).iterdir() if p.name != "Dockerfile"]
            assert len(staged) == 1
            assert dockerfile.count(f"COPY {staged[0]} ") == 2
        finally:
            import shutil

            shutil.rmtree(context_dir, ignore_errors=True)
