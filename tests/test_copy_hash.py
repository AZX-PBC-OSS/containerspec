"""Unit tests for copy() content hashing and validation errors. No Docker required."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestCopyContentHash:
    @pytest.mark.asyncio
    async def test_copy_content_change_busts_cache(self, tmp_path: Path) -> None:
        """Changing a copied file's content produces a different hash -> different tag."""
        from containerspec import ImageSpec

        app_py = tmp_path / "app.py"
        app_py.write_text("print('v1')")
        spec_v1 = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .copy(str(app_py), "/app/app.py")
            .entrypoint([])
        )
        tag_v1 = spec_v1.tag("test-copy", client=None)

        app_py.write_text("print('v2')")
        spec_v2 = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .copy(str(app_py), "/app/app.py")
            .entrypoint([])
        )
        tag_v2 = spec_v2.tag("test-copy", client=None)

        assert tag_v1 != tag_v2, "Different file content should produce different tags"


class TestCopyUserHash:
    def test_copy_with_explicit_hash(self) -> None:
        """copy() with content_hash= doesn't require file to exist."""
        from containerspec import ImageSpec

        spec = ImageSpec.from_registry("base", pin_digest=False).copy(
            "remote://file.py", "/app/file.py", content_hash="sha256:abc123"
        )
        assert spec.layers[0].content_hash == "sha256:abc123"

    def test_copy_explicit_hash_in_payload(self) -> None:
        """User-provided hash appears in canonical payload."""
        from containerspec import ImageSpec

        spec = ImageSpec.from_registry("base", pin_digest=False).copy(
            "remote://file.py", "/app/file.py", content_hash="sha256:abc123"
        )
        payload = spec._canonical_payload(client=None)
        copy_layer = next(layer for layer in payload["layers"] if layer["type"] == "copy")
        assert copy_layer["content_hash"] == "sha256:abc123"

    def test_copy_different_explicit_hashes_different_tag(self) -> None:
        """Different user-provided hashes produce different tags."""
        from containerspec import ImageSpec

        spec_a = ImageSpec.from_registry("base", pin_digest=False).copy(
            "file.py", "/app/file.py", content_hash="sha256:aaa"
        )
        spec_b = ImageSpec.from_registry("base", pin_digest=False).copy(
            "file.py", "/app/file.py", content_hash="sha256:bbb"
        )
        assert spec_a.tag("test", client=None) != spec_b.tag("test", client=None)


class TestValidationErrors:
    def test_copy_missing_file_no_hash_raises(self) -> None:
        """copy() with missing file and no content_hash raises FileNotFoundError."""
        from containerspec import ImageSpec

        with pytest.raises(FileNotFoundError, match="does not exist"):
            ImageSpec.from_registry("base", pin_digest=False).copy("/nonexistent", "/app")

    def test_build_error_has_context(self) -> None:
        """BuildError includes cmd and stderr for diagnostics."""
        from containerspec.backends import BuildError

        err = BuildError("test failed", cmd=["docker", "build"], stderr="some error")
        assert err.cmd == ["docker", "build"]
        assert err.stderr == "some error"
        assert "docker build" in str(err)
        assert "some error" in str(err)
