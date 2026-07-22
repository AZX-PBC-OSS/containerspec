"""Build targets — output formats for ImageSpec.build()."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BuiltImage:
    """Result of building a Docker target."""

    tag: str
    digest: str | None = None


@dataclass(frozen=True)
class FirecrackerRootfs:
    """Result of building a Firecracker rootfs target."""

    path: str
    hash: str
    size_mb: int


@dataclass(frozen=True)
class OciArtifact:
    """Result of building an OCI tarball target."""

    path: str
    hash: str


@runtime_checkable
class BuildTarget(Protocol):
    """A build output target. Implement to add new output formats."""

    @property
    def name(self) -> str: ...

    @property
    def needs_client(self) -> bool: ...

    def exists(self, *, hash: str, client: Any | None) -> bool: ...

    async def export(
        self,
        *,
        dockerfile: str,
        tag: str,
        canonical_json: str,
        client: Any | None,
        backend: Any,
        pull: bool,
    ) -> Any: ...

    def result_from_cache(self, *, hash: str, client: Any | None) -> Any: ...


def _sidecar_path(artifact_path: str) -> Path:
    return Path(f"{artifact_path}.containerspec.json")


def _sidecar_exists(artifact_path: str, hash_16: str) -> bool:
    """Check if a sidecar exists with a matching bare 16-char hash."""
    sidecar = _sidecar_path(artifact_path)
    if not sidecar.exists():
        return False
    try:
        data = json.loads(sidecar.read_text())
        return data.get("hash") == hash_16
    except (json.JSONDecodeError, KeyError):
        return False


@dataclass(frozen=True)
class DockerTarget:
    """Build and load into a Docker daemon."""

    name: str

    @property
    def needs_client(self) -> bool:
        return True

    def exists(self, *, hash: str, client: Any | None) -> bool:
        if client is None:
            return False
        tag = f"{self.name}:sha-{hash}"
        try:
            client.images.get(tag)
            return True
        except Exception:
            return False

    async def export(
        self,
        *,
        dockerfile: str,
        tag: str,
        canonical_json: str,
        client: Any | None,
        backend: Any,
        pull: bool,
    ) -> BuiltImage:
        await backend.solve_and_export(
            dockerfile=dockerfile,
            tag=tag,
            output_type="docker",
            output_path=None,
            labels={"containerspec.image_spec": canonical_json},
            pull=pull,
        )
        return BuiltImage(tag=tag)

    def result_from_cache(self, *, hash: str, client: Any | None) -> BuiltImage:
        tag = f"{self.name}:sha-{hash}"
        return BuiltImage(tag=tag)


@dataclass(frozen=True)
class FirecrackerRootfsTarget:
    """Build a Firecracker rootfs ext4 image. No Docker daemon needed (mke2fs mode).

    converter:
        - "mke2fs" (default): uses host mke2fs -d, no Docker daemon needed, requires e2fsprogs
        - "oci2rootfs": uses oci2rootfs in a container, requires Docker daemon, no host e2fsprogs.
          Handles full OCI whiteout semantics. converter_image specifies the container image.
    """

    path: str
    size_mb: int = 1024
    converter: str = "mke2fs"
    converter_image: str = "oci2rootfs:latest"

    @property
    def name(self) -> str:
        return "containerspec-rootfs"

    @property
    def needs_client(self) -> bool:
        return self.converter == "oci2rootfs"

    def exists(self, *, hash: str, client: Any | None) -> bool:
        return _sidecar_exists(self.path, hash)

    async def export(
        self,
        *,
        dockerfile: str,
        tag: str,
        canonical_json: str,
        client: Any | None,
        backend: Any,
        pull: bool,
    ) -> FirecrackerRootfs:
        import tempfile

        from containerspec.rootfs import write_sidecar

        hash_16 = tag.split("sha-")[-1]

        if self.converter == "oci2rootfs":
            from containerspec.rootfs import convert_oci_to_rootfs

            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as oci_tar:
                oci_tar_path = oci_tar.name
            try:
                await backend.solve_and_export(
                    dockerfile=dockerfile,
                    tag=tag,
                    output_type="oci",
                    output_path=oci_tar_path,
                    labels={},
                    pull=pull,
                )
                await convert_oci_to_rootfs(
                    oci_tar_path=oci_tar_path,
                    dest=self.path,
                    size_mb=self.size_mb,
                    converter_image=self.converter_image,
                )
            finally:
                from pathlib import Path as _Path

                _Path(oci_tar_path).unlink(missing_ok=True)
        else:
            from containerspec.rootfs import check_mke2fs, create_ext4

            check_mke2fs()
            with tempfile.TemporaryDirectory() as tmpdir:
                await backend.solve_and_export(
                    dockerfile=dockerfile,
                    tag=tag,
                    output_type="local",
                    output_path=tmpdir,
                    labels={},
                    pull=pull,
                )
                await create_ext4(rootfs_dir=tmpdir, dest=self.path, size_mb=self.size_mb)

        write_sidecar(
            f"{self.path}.containerspec.json",
            hash_16=hash_16,
            spec_json=canonical_json,
        )
        return FirecrackerRootfs(path=self.path, hash=hash_16, size_mb=self.size_mb)

    def result_from_cache(self, *, hash: str, client: Any | None) -> FirecrackerRootfs:
        return FirecrackerRootfs(path=self.path, hash=hash, size_mb=self.size_mb)


@dataclass(frozen=True)
class OciTarget:
    """Build an OCI tarball. No Docker daemon needed."""

    path: str

    @property
    def name(self) -> str:
        return "containerspec-oci"

    @property
    def needs_client(self) -> bool:
        return False

    def exists(self, *, hash: str, client: Any | None) -> bool:
        return _sidecar_exists(self.path, hash)

    async def export(
        self,
        *,
        dockerfile: str,
        tag: str,
        canonical_json: str,
        client: Any | None,
        backend: Any,
        pull: bool,
    ) -> OciArtifact:
        hash_16 = tag.split("sha-")[-1]
        await backend.solve_and_export(
            dockerfile=dockerfile,
            tag=tag,
            output_type="oci",
            output_path=self.path,
            labels={"containerspec.image_spec": canonical_json},
            pull=pull,
        )
        from containerspec.rootfs import write_sidecar

        write_sidecar(
            f"{self.path}.containerspec.json",
            hash_16=hash_16,
            spec_json=canonical_json,
        )
        return OciArtifact(path=self.path, hash=hash_16)

    def result_from_cache(self, *, hash: str, client: Any | None) -> OciArtifact:
        return OciArtifact(path=self.path, hash=hash)
