"""Rootfs creation — ext4 via mke2fs -d, sidecar metadata."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("containerspec")


class MissingToolError(RuntimeError):
    """Raised when a required host tool (e.g. mke2fs) is not found."""


def check_mke2fs() -> None:
    """Raise MissingToolError if mke2fs is not on PATH."""
    path = shutil.which("mke2fs")
    if path is None:
        raise MissingToolError(
            "mke2fs not found — install e2fsprogs to build Firecracker rootfs images"
        )
    logger.debug("containerspec.tool_check.mke2fs", extra={"path": path})


def check_buildah() -> None:
    """Raise MissingToolError if buildah is not on PATH."""
    path = shutil.which("buildah")
    if path is None:
        raise MissingToolError("buildah not found — install buildah to use the BuildahBackend")
    logger.debug("containerspec.tool_check.buildah", extra={"path": path})


async def create_ext4(*, rootfs_dir: str, dest: str, size_mb: int) -> None:
    """Create an ext4 image at dest, populated from rootfs_dir. No root needed."""
    check_mke2fs()
    size_arg = f"{size_mb}M"
    logger.info(
        "containerspec.rootfs.create",
        extra={"dest": dest, "size_mb": size_mb, "rootfs_dir": rootfs_dir},
    )
    proc = await asyncio.create_subprocess_exec(
        "mke2fs",
        "-t",
        "ext4",
        "-d",
        rootfs_dir,
        dest,
        size_arg,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"mke2fs failed with exit code {proc.returncode}:\n"
            f"{stderr.decode() if stderr else '(no stderr)'}"
        )
    logger.info("containerspec.rootfs.complete", extra={"dest": dest})


def write_sidecar(path: str, *, hash_16: str, spec_json: str) -> None:
    """Write sidecar metadata atomically (unique temp file + rename)."""
    data = {"hash": hash_16, "spec": json.loads(spec_json)}
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", prefix=".containerspec-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise
    logger.debug("containerspec.sidecar.written", extra={"path": path, "hash": hash_16})


def read_sidecar(path: str) -> dict[str, Any] | None:
    """Read sidecar metadata. Returns None if missing or invalid."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


async def convert_oci_to_rootfs(
    *,
    oci_tar_path: str,
    dest: str,
    size_mb: int,
    converter_image: str = "oci2rootfs:latest",
) -> None:
    """Convert an OCI tarball to an ext4 rootfs using oci2rootfs in a container.

    Isolates the conversion — no host e2fsprogs dependency. Requires Docker.
    The converter_image must have oci2rootfs CLI installed. Build one from
    https://github.com/arcboxlabs/oci2rootfs or provide your own.
    """
    import tempfile

    size_str = f"{size_mb}M"
    work_dir = tempfile.mkdtemp(prefix="containerspec-oci2rootfs-")
    try:
        oci_dir = f"{work_dir}/oci"
        output_dir = f"{work_dir}/output"
        Path(oci_dir).mkdir()
        Path(output_dir).mkdir()

        extract_cmd = ["tar", "xf", oci_tar_path, "-C", oci_dir]
        proc = await asyncio.create_subprocess_exec(
            *extract_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"tar extract failed with exit code {proc.returncode}:\n"
                f"{stderr.decode() if stderr else '(no stderr)'}"
            )

        convert_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{oci_dir}:/oci:ro",
            "-v",
            f"{output_dir}:/output",
            converter_image,
            "/oci",
            "--output",
            "/output/rootfs.ext4",
            "--size",
            size_str,
        ]
        logger.info("containerspec.oci2rootfs.run", extra={"converter_image": converter_image})
        proc = await asyncio.create_subprocess_exec(
            *convert_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"oci2rootfs failed with exit code {proc.returncode}:\n"
                f"{stderr.decode() if stderr else '(no stderr)'}"
            )

        import shutil as _shutil

        _shutil.move(f"{output_dir}/rootfs.ext4", dest)
        logger.info("containerspec.oci2rootfs.complete", extra={"dest": dest})
    finally:
        import shutil as _shutil

        _shutil.rmtree(work_dir, ignore_errors=True)
