from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from containerspec import (
    BuiltImage,
    DockerTarget,
    FirecrackerRootfs,
    FirecrackerRootfsTarget,
    OciArtifact,
    OciTarget,
)


class TestDockerTarget:
    def test_name(self) -> None:
        assert DockerTarget(name="warden/vllm").name == "warden/vllm"

    def test_needs_client_true(self) -> None:
        assert DockerTarget(name="x").needs_client is True

    def test_exists_true_when_image_in_daemon(self) -> None:
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        target = DockerTarget(name="warden/test")
        assert target.exists(hash="abc123", client=client) is True

    def test_exists_false_when_not_found(self) -> None:
        client = MagicMock()
        client.images.get.side_effect = Exception("not found")
        target = DockerTarget(name="warden/test")
        assert target.exists(hash="abc123", client=client) is False

    def test_result_from_cache(self) -> None:
        target = DockerTarget(name="warden/test")
        result = target.result_from_cache(hash="abc123def4567890", client=None)
        assert isinstance(result, BuiltImage)
        assert result.tag == "warden/test:sha-abc123def4567890"


class TestFirecrackerRootfsTarget:
    def test_defaults(self) -> None:
        t = FirecrackerRootfsTarget(path="/tmp/rootfs.ext4")
        assert t.size_mb == 1024

    def test_needs_client_false(self) -> None:
        assert FirecrackerRootfsTarget(path="/tmp/x").needs_client is False

    def test_exists_false_when_no_sidecar(self, tmp_path: Path) -> None:
        target = FirecrackerRootfsTarget(path=str(tmp_path / "rootfs.ext4"))
        assert target.exists(hash="abc123def4567890", client=None) is False

    def test_exists_true_when_sidecar_hash_matches(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        rootfs_path.write_bytes(b"fake ext4")
        sidecar = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar.write_text(json.dumps({"hash": "abc123def4567890", "spec": {}}))
        target = FirecrackerRootfsTarget(path=str(rootfs_path))
        assert target.exists(hash="abc123def4567890", client=None) is True

    def test_exists_false_when_sidecar_hash_mismatch(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        rootfs_path.write_bytes(b"fake ext4")
        sidecar = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar.write_text(json.dumps({"hash": "differenthash1234", "spec": {}}))
        target = FirecrackerRootfsTarget(path=str(rootfs_path))
        assert target.exists(hash="abc123def4567890", client=None) is False

    def test_result_from_cache(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        rootfs_path.write_bytes(b"fake")
        target = FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=2048)
        result = target.result_from_cache(hash="abc123def4567890", client=None)
        assert isinstance(result, FirecrackerRootfs)
        assert result.path == str(rootfs_path)
        assert result.hash == "abc123def4567890"
        assert result.size_mb == 2048


class TestFirecrackerRootfsConverter:
    def test_default_converter_is_mke2fs(self) -> None:
        t = FirecrackerRootfsTarget(path="/tmp/rootfs.ext4")
        assert t.converter == "mke2fs"

    def test_oci2rootfs_converter(self) -> None:
        t = FirecrackerRootfsTarget(path="/tmp/rootfs.ext4", converter="oci2rootfs")
        assert t.converter == "oci2rootfs"
        assert t.converter_image == "oci2rootfs:latest"

    def test_oci2rootfs_needs_client(self) -> None:
        t = FirecrackerRootfsTarget(path="/tmp/rootfs.ext4", converter="oci2rootfs")
        assert t.needs_client is True

    def test_mke2fs_does_not_need_client(self) -> None:
        t = FirecrackerRootfsTarget(path="/tmp/rootfs.ext4", converter="mke2fs")
        assert t.needs_client is False

    def test_custom_converter_image(self) -> None:
        t = FirecrackerRootfsTarget(
            path="/tmp/rootfs.ext4",
            converter="oci2rootfs",
            converter_image="myreg/oci2rootfs:v2",
        )
        assert t.converter_image == "myreg/oci2rootfs:v2"


class TestFirecrackerRootfsExport:
    @pytest.mark.asyncio
    async def test_export_extracts_bare_hash_and_writes_sidecar(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        target = FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=64)
        backend = AsyncMock()
        with (
            patch("containerspec.rootfs.check_mke2fs"),
            patch("containerspec.rootfs.create_ext4", new_callable=AsyncMock),
        ):
            result = await target.export(
                dockerfile="FROM base\n",
                tag="containerspec-rootfs:sha-abc123def4567890",
                canonical_json='{"base": {}}',
                client=None,
                backend=backend,
                pull=False,
            )
        backend.solve_and_export.assert_awaited_once()
        kwargs = backend.solve_and_export.call_args.kwargs
        assert kwargs["output_type"] == "local"
        assert kwargs["pull"] is False
        assert kwargs["labels"] == {}
        # Sidecar stores the bare 16-char hash (no "sha-" prefix, no full tag).
        sidecar = json.loads((tmp_path / "rootfs.ext4.containerspec.json").read_text())
        assert sidecar["hash"] == "abc123def4567890"
        assert "sha-" not in sidecar["hash"]
        # Result carries the bare hash, consistent with result_from_cache.
        assert isinstance(result, FirecrackerRootfs)
        assert result.hash == "abc123def4567890"
        assert result.path == str(rootfs_path)
        assert result.size_mb == 64

    @pytest.mark.asyncio
    async def test_oci2rootfs_export_invokes_converter_and_writes_sidecar(
        self, tmp_path: Path
    ) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        target = FirecrackerRootfsTarget(
            path=str(rootfs_path),
            size_mb=128,
            converter="oci2rootfs",
            converter_image="myreg/oci2rootfs:v2",
        )
        backend = AsyncMock()
        with patch(
            "containerspec.rootfs.convert_oci_to_rootfs", new_callable=AsyncMock
        ) as mock_conv:
            result = await target.export(
                dockerfile="FROM base\n",
                tag="containerspec-rootfs:sha-abc123def4567890",
                canonical_json='{"base": {}}',
                client=None,
                backend=backend,
                pull=True,
            )
        backend.solve_and_export.assert_awaited_once()
        solve_kwargs = backend.solve_and_export.call_args.kwargs
        assert solve_kwargs["output_type"] == "oci"
        assert solve_kwargs["pull"] is True
        assert solve_kwargs["labels"] == {}
        mock_conv.assert_awaited_once()
        conv_kwargs = mock_conv.call_args.kwargs
        assert conv_kwargs["dest"] == str(rootfs_path)
        assert conv_kwargs["size_mb"] == 128
        assert conv_kwargs["converter_image"] == "myreg/oci2rootfs:v2"
        sidecar = json.loads((tmp_path / "rootfs.ext4.containerspec.json").read_text())
        assert sidecar["hash"] == "abc123def4567890"
        assert isinstance(result, FirecrackerRootfs)
        assert result.hash == "abc123def4567890"
        assert result.size_mb == 128


class TestOciTarget:
    def test_needs_client_false(self) -> None:
        assert OciTarget(path="/tmp/tar.tar").needs_client is False

    def test_exists_false_when_no_sidecar(self, tmp_path: Path) -> None:
        target = OciTarget(path=str(tmp_path / "tar.tar"))
        assert target.exists(hash="abc123", client=None) is False

    def test_exists_true_when_sidecar_hash_matches(self, tmp_path: Path) -> None:
        oci_path = tmp_path / "tar.tar"
        oci_path.write_bytes(b"fake tar")
        sidecar = tmp_path / "tar.tar.containerspec.json"
        sidecar.write_text(json.dumps({"hash": "abc123def4567890", "spec": {}}))
        target = OciTarget(path=str(oci_path))
        assert target.exists(hash="abc123def4567890", client=None) is True

    def test_result_from_cache(self, tmp_path: Path) -> None:
        oci_path = tmp_path / "tar.tar"
        oci_path.write_bytes(b"fake tar")
        target = OciTarget(path=str(oci_path))
        result = target.result_from_cache(hash="abc123def4567890", client=None)
        assert isinstance(result, OciArtifact)
        assert result.path == str(oci_path)
        assert result.hash == "abc123def4567890"


class TestOciExport:
    @pytest.mark.asyncio
    async def test_export_passes_labels_and_writes_sidecar(self, tmp_path: Path) -> None:
        oci_path = tmp_path / "tar.tar"
        target = OciTarget(path=str(oci_path))
        backend = AsyncMock()
        result = await target.export(
            dockerfile="FROM base\n",
            tag="containerspec-oci:sha-abc123def4567890",
            canonical_json='{"base": {}}',
            client=None,
            backend=backend,
            pull=True,
        )
        kwargs = backend.solve_and_export.call_args.kwargs
        assert kwargs["output_type"] == "oci"
        assert kwargs["output_path"] == str(oci_path)
        # OCI target passes labels (not an empty dict).
        assert kwargs["labels"] == {"containerspec.image_spec": '{"base": {}}'}
        assert kwargs["pull"] is True
        sidecar = json.loads((tmp_path / "tar.tar.containerspec.json").read_text())
        assert sidecar["hash"] == "abc123def4567890"
        assert isinstance(result, OciArtifact)
        assert result.hash == "abc123def4567890"
        assert result.path == str(oci_path)


class TestResultTypes:
    def test_built_image_fields(self) -> None:
        r = BuiltImage(tag="x:sha-abc")
        assert r.tag == "x:sha-abc"
        assert r.digest is None

        r2 = BuiltImage(tag="x:sha-abc", digest="sha256:xyz")
        assert r2.digest == "sha256:xyz"

    def test_firecracker_rootfs_fields(self) -> None:
        r = FirecrackerRootfs(path="/tmp/rootfs.ext4", hash="abc", size_mb=2048)
        assert r.path == "/tmp/rootfs.ext4"
        assert r.hash == "abc"
        assert r.size_mb == 2048

    def test_oci_artifact_fields(self) -> None:
        r = OciArtifact(path="/tmp/tar.tar", hash="abc")
        assert r.path == "/tmp/tar.tar"
        assert r.hash == "abc"
