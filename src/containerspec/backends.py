"""Build backends — BuildKitBackend (docker buildx), BuildahBackend (daemonless), DockerBackend (docker-py)."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("containerspec")


class BuildError(RuntimeError):
    """Raised when a build subprocess fails. Includes stderr and command for diagnostics."""

    def __init__(
        self, message: str, *, cmd: list[str] | None = None, stderr: str | None = None
    ) -> None:
        self.cmd = cmd
        self.stderr = stderr
        parts = [message]
        if cmd:
            parts.append(f"Command: {' '.join(cmd)}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        super().__init__("\n".join(parts))


@runtime_checkable
class BuildBackend(Protocol):
    """A build execution backend."""

    async def solve_and_export(
        self,
        *,
        dockerfile: str,
        tag: str,
        output_type: str,
        output_path: str | None,
        labels: dict[str, str],
        pull: bool,
        context_path: str = ".",
    ) -> None: ...


def _write_dockerfile_temp(dockerfile: str) -> str:
    """Write Dockerfile to a temp file. Returns the path. Caller must clean up."""
    fd, path = tempfile.mkstemp(suffix="Dockerfile", prefix="containerspec-")
    with open(fd, "w") as f:
        f.write(dockerfile)
    logger.debug("containerspec.dockerfile.temp", extra={"path": path})
    return path


async def _run_command(cmd: list[str], *, label: str) -> tuple[int, bytes, bytes]:
    """Run a command, log it, return (rc, stdout, stderr)."""
    logger.info(f"containerspec.{label}.run", extra={"cmd": cmd})
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode() if stderr else "(no stderr)"
        logger.error(f"containerspec.{label}.failed", extra={"rc": proc.returncode, "stderr": err})
        raise BuildError(
            f"{label} failed with exit code {proc.returncode}",
            cmd=cmd,
            stderr=err,
        )
    logger.info(f"containerspec.{label}.complete")
    return proc.returncode, stdout, stderr


@dataclass(frozen=True)
class BuildKitBackend:
    """Build via ``docker buildx build`` CLI. Supports all output types.

    Dockerfile is written to a temp file and passed via ``-f`` — more robust
    than stdin piping, and the temp file is preserved on failure for debugging.
    """

    url: str | None = None
    builder: str | None = None

    async def solve_and_export(
        self,
        *,
        dockerfile: str,
        tag: str,
        output_type: str,
        output_path: str | None,
        labels: dict[str, str],
        pull: bool,
        context_path: str = ".",
    ) -> None:
        dockerfile_path = _write_dockerfile_temp(dockerfile)
        try:
            cmd: list[str] = ["docker", "buildx", "build"]
            if self.builder:
                cmd.extend(["--builder", self.builder])
            if output_path:
                output_val = f"type={output_type},dest={output_path}"
            else:
                output_val = f"type={output_type}"
            cmd.extend(["--output", output_val, "--tag", tag, "-f", dockerfile_path])
            if pull:
                cmd.append("--pull")
            for k, v in labels.items():
                cmd.extend(["--label", f"{k}={v}"])
            cmd.append(context_path)
            await _run_command(cmd, label="buildx")
        finally:
            Path(dockerfile_path).unlink(missing_ok=True)


@dataclass(frozen=True)
class BuildahBackend:
    """Build via ``buildah bud`` CLI. Daemonless — no Docker daemon needed.

    Default for FirecrackerRootfsTarget and OciTarget. Does NOT support
    ``output_type="docker"`` (use BuildKitBackend or DockerBackend for that).
    Linux-only — ``MissingToolError`` if buildah is not on PATH.
    """

    async def solve_and_export(
        self,
        *,
        dockerfile: str,
        tag: str,
        output_type: str,
        output_path: str | None,
        labels: dict[str, str],
        pull: bool,
        context_path: str = ".",
    ) -> None:
        from containerspec.rootfs import check_buildah

        check_buildah()

        if output_type == "docker":
            raise BuildError(
                "BuildahBackend does not support output_type='docker'. "
                "Use BuildKitBackend or DockerBackend for Docker targets."
            )

        dockerfile_path = _write_dockerfile_temp(dockerfile)
        try:
            build_cmd: list[str] = ["buildah", "bud", "-f", dockerfile_path, "-t", tag]
            if pull:
                build_cmd.append("--pull")
            for k, v in labels.items():
                build_cmd.extend(["--label", f"{k}={v}"])
            build_cmd.append(context_path)
            await _run_command(build_cmd, label="buildah.bud")

            if output_type == "oci" and output_path:
                push_cmd = ["buildah", "push", "-f", "oci-archive", tag, output_path]
                await _run_command(push_cmd, label="buildah.push.oci")
            elif output_type == "local" and output_path:
                await self._export_filesystem(tag=tag, dest=output_path)
        finally:
            Path(dockerfile_path).unlink(missing_ok=True)

    async def _export_filesystem(self, *, tag: str, dest: str) -> None:
        """Export the working container's filesystem to dest using buildah mount."""
        from_cmd = ["buildah", "from", tag]
        _, stdout, _ = await _run_command(from_cmd, label="buildah.from")
        container_name = stdout.decode().strip()

        try:
            mount_cmd = ["buildah", "mount", container_name]
            _, stdout, _ = await _run_command(mount_cmd, label="buildah.mount")
            mountpoint = stdout.decode().strip()

            copy_cmd = ["cp", "-a", f"{mountpoint}/.", dest + "/"]
            await _run_command(copy_cmd, label="buildah.copy")

            umount_cmd = ["buildah", "umount", container_name]
            await _run_command(umount_cmd, label="buildah.umount")
        finally:
            rm_cmd = ["buildah", "rm", container_name]
            await _run_command(rm_cmd, label="buildah.rm")


@dataclass
class DockerBackend:
    """Build via docker-py. Docker target only (fallback when buildx unavailable)."""

    client: Any = None

    async def solve_and_export(
        self,
        *,
        dockerfile: str,
        tag: str,
        output_type: str,
        output_path: str | None,
        labels: dict[str, str],
        pull: bool,
        context_path: str = ".",
    ) -> None:
        if output_type != "docker":
            raise BuildError(
                f"DockerBackend does not support output_type='{output_type}'. "
                "Use BuildKitBackend or BuildahBackend for non-Docker targets."
            )
        import io

        if self.client is None:
            import docker

            self.client = docker.from_env()
        logger.info("containerspec.docker.build", extra={"tag": tag})
        build_kwargs: dict[str, Any] = {
            "tag": tag,
            "pull": pull,
            "rm": True,
            "labels": labels,
        }
        if context_path and context_path != ".":
            # The staged context is self-contained: _prepare_build_context wrote
            # the rewritten Dockerfile into it, and the daemon resolves
            # ``dockerfile`` relative to the context.
            build_kwargs["dockerfile"] = "Dockerfile"
            build_kwargs["path"] = context_path
            await asyncio.to_thread(self.client.images.build, **build_kwargs)
        else:
            build_kwargs["fileobj"] = io.BytesIO(dockerfile.encode())
            await asyncio.to_thread(self.client.images.build, **build_kwargs)
        logger.info("containerspec.docker.complete", extra={"tag": tag})


def auto_detect_backend(*, target: Any = None) -> BuildBackend:
    """Pick the best backend for the given target.

    - DockerTarget: BuildKitBackend if buildx is available, else DockerBackend.
    - FirecrackerRootfsTarget / OciTarget: BuildahBackend if available (daemonless),
      else BuildKitBackend if buildx is available, else DockerBackend (will error
      for non-Docker output types).
    """
    from containerspec.targets import DockerTarget

    is_docker_target = isinstance(target, DockerTarget) if target else True

    if is_docker_target:
        if shutil.which("docker") is not None:
            return BuildKitBackend()
        return DockerBackend()

    if shutil.which("buildah") is not None:
        return BuildahBackend()
    if shutil.which("docker") is not None:
        return BuildKitBackend()
    return DockerBackend()
