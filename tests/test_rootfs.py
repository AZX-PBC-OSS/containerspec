from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from containerspec.rootfs import (
    MissingToolError,
    check_buildah,
    check_mke2fs,
    convert_oci_to_rootfs,
    create_ext4,
    read_sidecar,
    write_sidecar,
)


class TestCheckMke2fs:
    def test_raises_when_mke2fs_not_found(self) -> None:
        with (
            patch("containerspec.rootfs.shutil.which", return_value=None),
            pytest.raises(MissingToolError, match="mke2fs not found"),
        ):
            check_mke2fs()

    def test_passes_when_mke2fs_found(self) -> None:
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/sbin/mke2fs"):
            check_mke2fs()


class TestCheckBuildah:
    def test_raises_when_buildah_not_found(self) -> None:
        with (
            patch("containerspec.rootfs.shutil.which", return_value=None),
            pytest.raises(MissingToolError, match="buildah not found"),
        ):
            check_buildah()

    def test_passes_when_buildah_found(self) -> None:
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"):
            check_buildah()


class TestCreateExt4:
    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_calls_mke2fs_with_correct_args(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate.return_value = (b"", b"")
        mock_exec.return_value = proc
        with patch("containerspec.rootfs.check_mke2fs"):
            await create_ext4(rootfs_dir="/tmp/rootfs", dest="/tmp/rootfs.ext4", size_mb=512)
        cmd = list(mock_exec.call_args.args)
        assert "mke2fs" in cmd[0]
        assert "-t" in cmd
        assert "ext4" in cmd
        assert "-d" in cmd
        assert "/tmp/rootfs" in cmd
        assert "/tmp/rootfs.ext4" in cmd
        assert "512M" in cmd

    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_raises_on_nonzero_exit_includes_stderr(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate.return_value = (b"", b"mke2fs: bad blocks")
        mock_exec.return_value = proc
        with (
            patch("containerspec.rootfs.check_mke2fs"),
            pytest.raises(RuntimeError, match="mke2fs failed"),
        ):
            await create_ext4(rootfs_dir="/tmp/rootfs", dest="/tmp/rootfs.ext4", size_mb=512)

    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_error_message_contains_stderr(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.returncode = 2
        proc.communicate.return_value = (b"", b"disk full error")
        mock_exec.return_value = proc
        with (
            patch("containerspec.rootfs.check_mke2fs"),
            pytest.raises(RuntimeError, match="disk full error"),
        ):
            await create_ext4(rootfs_dir="/tmp/rootfs", dest="/tmp/rootfs.ext4", size_mb=512)


class TestWriteSidecar:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "rootfs.ext4.containerspec.json"
        write_sidecar(str(sidecar_path), hash_16="abc123def4567890", spec_json='{"base": {}}')
        data = json.loads(sidecar_path.read_text())
        assert data["hash"] == "abc123def4567890"
        assert data["spec"] == {"base": {}}

    def test_atomic_write(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "rootfs.ext4.containerspec.json"
        write_sidecar(str(sidecar_path), hash_16="abc123def4567890", spec_json="{}")
        assert sidecar_path.exists()
        assert not (tmp_path / "rootfs.ext4.containerspec.json.tmp").exists()

    def test_stores_bare_hash_not_tag(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "rootfs.ext4.containerspec.json"
        write_sidecar(str(sidecar_path), hash_16="0123456789abcdef", spec_json="{}")
        data = json.loads(sidecar_path.read_text())
        # Bare 16-char hash — no "sha-" prefix, no full tag.
        assert data["hash"] == "0123456789abcdef"
        assert "sha-" not in data["hash"]


class TestReadSidecar:
    def test_reads_valid_sidecar(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar_path.write_text(json.dumps({"hash": "abc123def4567890", "spec": {"base": {}}}))
        data = read_sidecar(str(sidecar_path))
        assert data["hash"] == "abc123def4567890"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert read_sidecar(str(tmp_path / "nonexistent.json")) is None

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "bad.json"
        sidecar_path.write_text("not json")
        assert read_sidecar(str(sidecar_path)) is None


class TestConvertOciToRootfs:
    @patch("shutil.move")
    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_runs_docker_with_oci2rootfs(
        self,
        mock_exec: MagicMock,
        mock_move: MagicMock,
    ) -> None:
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate.return_value = (b"", b"")
        mock_exec.return_value = proc
        await convert_oci_to_rootfs(
            oci_tar_path="/tmp/oci.tar",
            dest="/tmp/rootfs.ext4",
            size_mb=512,
        )
        assert mock_exec.call_count == 2
        docker_cmd = list(mock_exec.call_args_list[1].args)
        assert docker_cmd[0] == "docker"
        assert "run" in docker_cmd
        assert "--rm" in docker_cmd
        assert "oci2rootfs:latest" in docker_cmd
        assert "/oci" in docker_cmd
        assert "--output" in docker_cmd
        assert "/output/rootfs.ext4" in docker_cmd
        assert "--size" in docker_cmd
        assert "512M" in docker_cmd
        mock_move.assert_called_once()

    @patch("shutil.move")
    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_extracts_oci_tarball_first(
        self,
        mock_exec: MagicMock,
        mock_move: MagicMock,
    ) -> None:
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate.return_value = (b"", b"")
        mock_exec.return_value = proc
        await convert_oci_to_rootfs(
            oci_tar_path="/tmp/oci.tar",
            dest="/tmp/rootfs.ext4",
            size_mb=256,
        )
        first_cmd = list(mock_exec.call_args_list[0].args)
        assert first_cmd[0] == "tar"
        assert "xf" in first_cmd
        assert "/tmp/oci.tar" in first_cmd
        assert "-C" in first_cmd

    @patch("shutil.move")
    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_raises_on_nonzero_docker_exit(
        self,
        mock_exec: MagicMock,
        mock_move: MagicMock,
    ) -> None:
        proc_ok = AsyncMock()
        proc_ok.returncode = 0
        proc_ok.communicate.return_value = (b"", b"")
        proc_fail = AsyncMock()
        proc_fail.returncode = 1
        proc_fail.communicate.return_value = (b"", b"oci2rootfs boom")
        mock_exec.side_effect = [proc_ok, proc_fail]
        with pytest.raises(RuntimeError, match="oci2rootfs failed"):
            await convert_oci_to_rootfs(
                oci_tar_path="/tmp/oci.tar",
                dest="/tmp/rootfs.ext4",
                size_mb=512,
            )
        mock_move.assert_not_called()

    @patch("shutil.move")
    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_tar_extract_failure_raises(
        self,
        mock_exec: MagicMock,
        mock_move: MagicMock,
    ) -> None:
        proc_fail = AsyncMock()
        proc_fail.returncode = 1
        proc_fail.communicate.return_value = (b"", b"tar: bad archive")
        mock_exec.return_value = proc_fail
        with pytest.raises(RuntimeError, match="tar extract failed"):
            await convert_oci_to_rootfs(
                oci_tar_path="/tmp/oci.tar",
                dest="/tmp/rootfs.ext4",
                size_mb=512,
            )
        assert mock_exec.call_count == 1
        mock_move.assert_not_called()

    @patch("shutil.move")
    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_converter_image_passed_through(
        self,
        mock_exec: MagicMock,
        mock_move: MagicMock,
    ) -> None:
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate.return_value = (b"", b"")
        mock_exec.return_value = proc
        await convert_oci_to_rootfs(
            oci_tar_path="/tmp/oci.tar",
            dest="/tmp/rootfs.ext4",
            size_mb=64,
            converter_image="myreg/oci2rootfs:v2",
        )
        docker_cmd = list(mock_exec.call_args_list[1].args)
        assert "myreg/oci2rootfs:v2" in docker_cmd
