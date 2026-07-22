from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from containerspec import (
    BuiltImage,
    FirecrackerRootfs,
    FirecrackerRootfsTarget,
    ImageSpec,
    OciArtifact,
    OciTarget,
)


class TestBuildStringShortcut:
    @pytest.mark.asyncio
    async def test_string_target_normalizes_to_docker_target_cache_hit(self) -> None:
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        spec = (
            ImageSpec.from_registry("python:3.12", pin_digest=False)
            .env({"HF_HOME": "/data/hf"})
            .user(uid=1000, gid=1000, name="app")
        )
        hash_16 = spec.content_hash(client=client)[:16]
        backend = AsyncMock()

        result = await spec.build("warden/test", client=client, backend=backend)

        assert isinstance(result, BuiltImage)
        assert result.tag == f"warden/test:sha-{hash_16}"
        client.images.get.assert_called_once_with(f"warden/test:sha-{hash_16}")
        backend.solve_and_export.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_string_target_cache_miss_dispatches_to_backend(self) -> None:
        client = MagicMock()
        client.images.get.side_effect = Exception("not found")
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False)
        hash_16 = spec.content_hash(client=client)[:16]
        backend = AsyncMock()

        result = await spec.build("warden/test", client=client, backend=backend)

        assert isinstance(result, BuiltImage)
        assert result.tag == f"warden/test:sha-{hash_16}"
        backend.solve_and_export.assert_awaited_once()
        kwargs = backend.solve_and_export.call_args.kwargs
        assert kwargs["output_type"] == "docker"
        assert kwargs["output_path"] is None
        assert kwargs["tag"] == f"warden/test:sha-{hash_16}"
        assert kwargs["pull"] is True


class TestBuildStagedContextHandoff:
    @pytest.mark.asyncio
    async def test_backend_receives_rewritten_dockerfile_for_copy_layers(
        self, tmp_path: Path
    ) -> None:
        """Backends must get the context-rewritten Dockerfile, not the original.

        The staged context contains copied sources under ``ctx_*`` names; a
        Dockerfile still COPYing the original paths cannot build from it.
        """
        src_file = tmp_path / "app.py"
        src_file.write_text("print('hi')\n")
        client = MagicMock()
        client.images.get.side_effect = Exception("not found")
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).copy(
            str(src_file), "/app/app.py"
        )
        backend = AsyncMock()

        await spec.build("warden/test", client=client, backend=backend)

        kwargs = backend.solve_and_export.call_args.kwargs
        assert kwargs["context_path"] != "."
        assert f"COPY {src_file}" not in kwargs["dockerfile"]
        assert "COPY ctx_" in kwargs["dockerfile"]


class TestBuildRootfsTarget:
    @pytest.mark.asyncio
    async def test_dispatches_to_backend_with_local_output(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        target = FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=64)
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False)
        hash_16 = spec.content_hash(client=None)[:16]
        backend = AsyncMock()

        with (
            patch("containerspec.rootfs.check_mke2fs"),
            patch("containerspec.rootfs.create_ext4", new_callable=AsyncMock),
        ):
            result = await spec.build(target, client=None, backend=backend)

        backend.solve_and_export.assert_awaited_once()
        kwargs = backend.solve_and_export.call_args.kwargs
        assert kwargs["output_type"] == "local"
        assert kwargs["output_path"] is not None
        assert kwargs["labels"] == {}
        assert kwargs["pull"] is True
        assert isinstance(result, FirecrackerRootfs)
        assert result.path == str(rootfs_path)
        assert result.hash == hash_16
        assert result.size_mb == 64

    @pytest.mark.asyncio
    async def test_cache_hit_skips_backend(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        rootfs_path.write_bytes(b"fake ext4")
        target = FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=64)
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False)
        hash_16 = spec.content_hash(client=None)[:16]
        sidecar = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar.write_text(json.dumps({"hash": hash_16, "spec": {}}))
        backend = AsyncMock()

        with (
            patch("containerspec.rootfs.check_mke2fs"),
            patch("containerspec.rootfs.create_ext4", new_callable=AsyncMock),
        ):
            result = await spec.build(target, client=None, backend=backend)

        backend.solve_and_export.assert_not_awaited()
        assert isinstance(result, FirecrackerRootfs)
        assert result.hash == hash_16
        assert result.path == str(rootfs_path)


class TestBuildOciTarget:
    @pytest.mark.asyncio
    async def test_dispatches_to_backend_with_oci_output(self, tmp_path: Path) -> None:
        oci_path = tmp_path / "tar.tar"
        target = OciTarget(path=str(oci_path))
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False)
        hash_16 = spec.content_hash(client=None)[:16]
        backend = AsyncMock()

        result = await spec.build(target, client=None, backend=backend)

        backend.solve_and_export.assert_awaited_once()
        kwargs = backend.solve_and_export.call_args.kwargs
        assert kwargs["output_type"] == "oci"
        assert kwargs["output_path"] == str(oci_path)
        assert "containerspec.image_spec" in kwargs["labels"]
        assert kwargs["pull"] is True
        assert isinstance(result, OciArtifact)
        assert result.path == str(oci_path)
        assert result.hash == hash_16


class TestResolveHfHome:
    def test_returns_hf_home_from_env_layer(self) -> None:
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).env({"HF_HOME": "/data/hf"})
        assert spec._resolve_hf_home() == "/data/hf"

    def test_returns_default_when_no_env_layer(self) -> None:
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False)
        assert spec._resolve_hf_home() == "/root/.cache/huggingface"

    def test_ignores_unrelated_env_vars(self) -> None:
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).env({"PATH": "/usr/bin"})
        assert spec._resolve_hf_home() == "/root/.cache/huggingface"


class TestResolveUid:
    def test_returns_uid_from_user_layer(self) -> None:
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False).user(
            uid=1000, gid=1000, name="app"
        )
        assert spec._resolve_uid() == 1000

    def test_returns_default_when_no_user_layer(self) -> None:
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False)
        assert spec._resolve_uid() == 0

    def test_returns_last_user_layer_uid(self) -> None:
        spec = (
            ImageSpec.from_registry("python:3.12", pin_digest=False)
            .user(uid=1000, gid=1000, name="app")
            .user(uid=2000, gid=2000, name="svc")
        )
        assert spec._resolve_uid() == 2000


class TestResolveDigest:
    def test_returns_none_when_pin_digest_false(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False)
        assert spec._resolve_digest(client=MagicMock()) is None

    def test_returns_digest_when_pin_digest_true(self) -> None:
        client = MagicMock()
        client.images.get_registry_data.return_value = MagicMock(id="sha256:abc123")
        spec = ImageSpec.from_registry("base", pin_digest=True)
        assert spec._resolve_digest(client=client) == "sha256:abc123"


class TestToBuildDockerfile:
    def test_pins_from_with_digest_when_pin_digest_true(self) -> None:
        client = MagicMock()
        client.images.get_registry_data.return_value = MagicMock(id="sha256:abc123")
        spec = ImageSpec.from_registry("base", pin_digest=True).uv_pip_install("vllm")
        df = spec._to_build_dockerfile(client=client)
        assert "FROM base@sha256:abc123" in df

    def test_no_digest_pin_when_pin_digest_false(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        df = spec._to_build_dockerfile(client=MagicMock())
        assert "FROM base\n" in df
        assert "@" not in df


class TestBuildClientAndBackendResolution:
    @pytest.mark.asyncio
    async def test_imports_docker_and_auto_detects_backend(self) -> None:
        spec = ImageSpec.from_registry("python:3.12", pin_digest=False)
        fake_client = MagicMock()
        fake_client.images.get.side_effect = Exception("not found")
        fake_docker = MagicMock()
        fake_docker.from_env.return_value = fake_client
        backend = AsyncMock()
        with (
            patch.dict(sys.modules, {"docker": fake_docker}),
            patch("containerspec.backends.auto_detect_backend", return_value=backend),
        ):
            result = await spec.build("warden/test", client=None, backend=None)

        fake_docker.from_env.assert_called_once()
        assert isinstance(result, BuiltImage)
        backend.solve_and_export.assert_awaited_once()
