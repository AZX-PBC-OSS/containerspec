"""ImageSpec — fluent, immutable image specification."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from containerspec.backends import BuildError
from containerspec.layers import (
    AddPython,
    ApkInstall,
    AptInstall,
    BrewInstall,
    CargoInstall,
    Chown,
    Cmd,
    Copy,
    CopyFromStage,
    DnfInstall,
    Entrypoint,
    Env,
    Expose,
    GemInstall,
    GoInstall,
    Layer,
    NpmInstall,
    NvmInstall,
    PacmanInstall,
    PipInstall,
    PnpmInstall,
    RunCommands,
    RustInstall,
    User,
    UvPipInstall,
    UvxInstall,
    Volume,
    Workdir,
    YarnInstall,
    ZypperInstall,
    frozen_mapping,
    layer_payload,
)
from containerspec.renderers import render_dockerfile
from containerspec.rootfs import MissingToolError
from containerspec.validation import validate_image_ref, validate_name

if TYPE_CHECKING:
    from containerspec.targets import BuildTarget

logger = logging.getLogger("containerspec")


@dataclass(frozen=True)
class StageSpec:
    """A named build stage for multi-stage Dockerfiles."""

    name: str
    spec: ImageSpec


def _hash_path(path: str) -> str:
    """Compute a recursive sha256 of a file or directory for cache busting."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"copy() source '{path}' does not exist. "
            f"The file must exist at spec construction time for content hashing."
        )
    h = hashlib.sha256()
    if p.is_file():
        h.update(p.read_bytes())
        return h.hexdigest()
    files = sorted(p.rglob("*"))
    for f in files:
        if f.is_file():
            rel = f.relative_to(p).as_posix()
            h.update(rel.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ImageSpec:
    """Fluent, immutable image specification.

    Build with ``ImageSpec.from_registry(base).apt_install(...).uv_pip_install(...)``
    and call ``.build(target)`` to produce a Docker image, Firecracker rootfs, or OCI tarball.

    For multi-stage builds, use ``copy_from_stage()`` to compose a build stage
    into a runtime image.
    """

    base: str
    pin_digest: bool = True
    layers: tuple[Layer, ...] = ()
    stages: tuple[StageSpec, ...] = ()
    distro: str | None = None

    @classmethod
    def from_registry(
        cls, base: str, *, pin_digest: bool = True, distro: str | None = None
    ) -> ImageSpec:
        """Create an ImageSpec from a base image.

        Args:
            base: Base image reference (e.g. ``"python:3.12-slim"``, ``"alpine:3.20"``).
            pin_digest: When True, resolve and pin the base image manifest digest in the hash.
            distro: Declare the base image's distro family for correct rendering before any
                package install. One of ``"debian"``, ``"alpine"``, ``"rhel"``. When None
                (default), inferred from the first ``.apt_install()``/``.apk_install()``/
                ``.dnf_install()`` call. Set explicitly if ``.user()`` precedes any package
                install on a non-Debian base (e.g. Alpine). An explicitly-set distro always
                participates in the content hash, so adding it to a previously-distro-less
                spec changes the cache key (one-time cache miss) even when rendering is
                identical.
        """
        if not base or not base.strip():
            raise ValueError("from_registry requires a non-empty base image reference")
        if distro is not None and distro not in (
            "debian",
            "alpine",
            "rhel",
            "fedora",
            "arch",
            "opensuse",
            "busybox",
        ):
            raise ValueError(
                f"distro must be one of 'debian', 'alpine', 'rhel', 'fedora', "
                f"'arch', 'opensuse', 'busybox' — got {distro!r}"
            )
        return cls(base=base, pin_digest=pin_digest, layers=(), stages=(), distro=distro)

    def _with(self, layer: Layer) -> ImageSpec:
        return replace(self, layers=(*self.layers, layer))

    def _with_stage(self, stage: StageSpec) -> ImageSpec:
        return replace(self, stages=(*self.stages, stage))

    def add_python(self, version: str) -> ImageSpec:
        return self._with(AddPython(version=version))

    def apt_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("apt_install requires at least one package")
        return self._with(AptInstall(packages=tuple(sorted(packages))))

    def apt_update(self) -> ImageSpec:
        """Run apt-get update && apt-get dist-upgrade -y to bring base image to latest."""
        return self._with(
            RunCommands(
                commands=(
                    "DEBIAN_FRONTEND=noninteractive apt-get update && "
                    "apt-get dist-upgrade -y --no-install-recommends && "
                    "apt-get clean && rm -rf /var/lib/apt/lists/*",
                )
            )
        )

    def apk_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("apk_install requires at least one package")
        return self._with(ApkInstall(packages=tuple(sorted(packages))))

    def dnf_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("dnf_install requires at least one package")
        return self._with(DnfInstall(packages=tuple(sorted(packages))))

    def dnf_update(self) -> ImageSpec:
        """Run dnf upgrade to bring base image to latest."""
        return self._with(RunCommands(commands=("dnf upgrade -y && dnf clean all",)))

    def pacman_install(self, *packages: str) -> ImageSpec:
        """Install packages via pacman (Arch Linux). Sorted within layer.

        Supports version pins (``pkg=1.2.3``), git AUR helpers, and exact versions.
        """
        if not packages:
            raise ValueError("pacman_install requires at least one package")
        return self._with(PacmanInstall(packages=tuple(sorted(packages))))

    def pacman_update(self) -> ImageSpec:
        """Run pacman -Syu to bring Arch base to latest."""
        return self._with(RunCommands(commands=("pacman -Syu --noconfirm",)))

    def aur_install(self, *packages: str) -> ImageSpec:
        """Install packages from the AUR (Arch User Repository).

        Uses yay as the AUR helper. Requires yay installed (via run_commands or base image).
        Sorted within layer for hash stability.
        """
        if not packages:
            raise ValueError("aur_install requires at least one package")
        return self._with(
            RunCommands(
                commands=(
                    " ".join(
                        f"yay -S --noconfirm {' '.join(sorted(packages))}",
                    ),
                )
            )
        )

    def zypper_install(self, *packages: str) -> ImageSpec:
        """Install packages via zypper (openSUSE). Sorted within layer."""
        if not packages:
            raise ValueError("zypper_install requires at least one package")
        return self._with(ZypperInstall(packages=tuple(sorted(packages))))

    def zypper_update(self) -> ImageSpec:
        """Run zypper update to bring openSUSE base to latest."""
        return self._with(RunCommands(commands=("zypper update -y",)))

    def yarn_install(self, *packages: str) -> ImageSpec:
        """Install npm packages globally via yarn. Requires nvm_install or base with node."""
        if not packages:
            raise ValueError("yarn_install requires at least one package")
        return self._with(YarnInstall(packages=tuple(sorted(packages))))

    def gem_install(self, *packages: str) -> ImageSpec:
        """Install Ruby gems. Requires Ruby in base image or installed via run_commands."""
        if not packages:
            raise ValueError("gem_install requires at least one package")
        return self._with(GemInstall(packages=tuple(sorted(packages))))

    def go_install(self, *packages: str) -> ImageSpec:
        """Install Go packages. Requires Go in base image or installed via run_commands.

        Supports version pins (``pkg@v1.2.3``) and git URLs.
        """
        if not packages:
            raise ValueError("go_install requires at least one package")
        return self._with(GoInstall(packages=tuple(sorted(packages))))

    def uv_pip_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("uv_pip_install requires at least one package")
        return self._with(UvPipInstall(packages=tuple(sorted(packages))))

    def pip_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("pip_install requires at least one package")
        return self._with(PipInstall(packages=tuple(sorted(packages))))

    def env(self, vars: Mapping[str, str]) -> ImageSpec:
        sorted_vars = {k: vars[k] for k in sorted(vars)}
        return self._with(Env(vars=frozen_mapping(sorted_vars)))

    def run_commands(self, *commands: str) -> ImageSpec:
        return self._with(RunCommands(commands=tuple(commands)))

    def workdir(self, path: str) -> ImageSpec:
        return self._with(Workdir(path=path))

    def chown(self, path: str, *, uid: int | None = None, gid: int | None = None) -> ImageSpec:
        return self._with(Chown(path=path, uid=uid, gid=gid))

    def user(self, *, uid: int, gid: int, name: str, alpine: bool = False) -> ImageSpec:
        """Create a non-root user.

        Args:
            uid: User ID.
            gid: Group ID.
            name: Username (also used for group name and home directory).
            alpine: Set True for Alpine/BusyBox-based images (uses addgroup/adduser -D
                instead of groupadd/useradd). Inferred automatically if .apk_install()
                precedes this layer, but set explicitly if .user() comes before any
                package install on an Alpine base.
        """
        return self._with(User(uid=uid, gid=gid, name=name, alpine=alpine))

    def entrypoint(self, commands: Sequence[str] | None) -> ImageSpec:
        normalized: tuple[str, ...] | None = tuple(commands) if commands is not None else None
        return self._with(Entrypoint(commands=normalized))

    def expose(self, *ports: int) -> ImageSpec:
        return self._with(Expose(ports=tuple(sorted(ports))))

    def cmd(self, commands: Sequence[str] | None) -> ImageSpec:
        normalized: tuple[str, ...] | None = tuple(commands) if commands is not None else None
        return self._with(Cmd(commands=normalized))

    def volume(self, *paths: str) -> ImageSpec:
        return self._with(Volume(paths=tuple(sorted(paths))))

    def copy(self, src: str, dest: str, *, content_hash: str | None = None) -> ImageSpec:
        """Copy a file or directory into the image (COPY).

        For cache busting, the source content is hashed and included in the
        canonical payload. Two modes:

        1. Local file exists: ``copy("./app.py", "/app/app.py")`` — hashes
           the file/directory content automatically.
        2. User provides hash: ``copy("remote://file", "/app/file", content_hash="sha256:...")``
           — for CI pipelines where the file isn't local but its hash is known.

        If the file doesn't exist AND no ``content_hash`` is provided, raises
        ``FileNotFoundError`` with a clear message.
        """
        if content_hash is not None:
            return self._with(Copy(src=src, dest=dest, content_hash=content_hash))
        try:
            computed = _hash_path(src)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"copy() source '{src}' does not exist. "
                f"Either provide a local path that exists, or pass content_hash= "
                f"explicitly: copy('{src}', '{dest}', content_hash='sha256:...')"
            ) from None
        return self._with(Copy(src=src, dest=dest, content_hash=computed))

    def copy_from_stage(self, stage: StageSpec, src: str, dest: str) -> ImageSpec:
        """Copy from a named build stage into this image (COPY --from=<stage>).

        The stage's content hash is included in the canonical payload, so changes
        to the stage's spec bust the cache. The stage is registered on first use;
        duplicate stage names are ignored so the Dockerfile emits a single
        ``FROM ... AS <name>`` per stage.
        """
        with_layer = self._with(
            CopyFromStage(
                stage_name=stage.name,
                stage_hash=stage.spec.content_hash(client=None),
                src=src,
                dest=dest,
            )
        )
        if any(s.name == stage.name for s in with_layer.stages):
            return with_layer
        return with_layer._with_stage(stage)

    def with_stage(self, name: str) -> StageSpec:
        """Create a named build stage from this spec for use in multi-stage builds."""
        return StageSpec(name=name, spec=self)

    def nvm_install(self, version: str) -> ImageSpec:
        """Install Node.js via nvm. Symlinks node/npm/npx to /usr/local/bin."""
        return self._with(NvmInstall(version=version))

    def npm_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("npm_install requires at least one package")
        return self._with(NpmInstall(packages=tuple(sorted(packages))))

    def pnpm_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("pnpm_install requires at least one package")
        return self._with(PnpmInstall(packages=tuple(sorted(packages))))

    def brew_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("brew_install requires at least one package")
        return self._with(BrewInstall(packages=tuple(sorted(packages))))

    def rust_install(self) -> ImageSpec:
        """Install Rust via rustup. Sets PATH for cargo/rustc."""
        return self._with(RustInstall())

    def cargo_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("cargo_install requires at least one package")
        return self._with(CargoInstall(packages=tuple(sorted(packages))))

    def uvx_install(self, *packages: str) -> ImageSpec:
        if not packages:
            raise ValueError("uvx_install requires at least one package")
        return self._with(UvxInstall(packages=tuple(sorted(packages))))

    def _canonical_payload(self, *, client: Any) -> dict[str, Any]:
        if self.pin_digest and client is not None:
            digest = client.images.get_registry_data(self.base).id
            base_entry: dict[str, Any] = {"ref": self.base, "digest": digest}
        elif self.pin_digest and client is None:
            logger.warning(
                "containerspec.hash.pin_digest_skipped",
                extra={"reason": "no client available for daemonless target, using tag string"},
            )
            base_entry: dict[str, Any] = {"ref": self.base}
        else:
            base_entry = {"ref": self.base}
        payload: dict[str, Any] = {
            "base": base_entry,
            "pin_digest": self.pin_digest,
            "layers": [layer_payload(layer) for layer in self.layers],
        }
        if self.distro:
            payload["distro"] = self.distro
        if self.stages:
            payload["stages"] = [
                {
                    "name": s.name,
                    "hash": s.spec.content_hash(client=client),
                }
                for s in self.stages
            ]
        return payload

    def _canonical_json(self, *, client: Any) -> str:
        import json

        return json.dumps(self._canonical_payload(client=client), sort_keys=True)

    def content_hash(self, *, client: Any = None) -> str:
        import hashlib

        return hashlib.sha256(self._canonical_json(client=client).encode()).hexdigest()

    def tag(self, name: str, *, client: Any = None) -> str:
        return f"{name}:sha-{self.content_hash(client=client)[:16]}"

    def _resolve_digest(self, *, client: Any) -> str | None:
        if not self.pin_digest:
            return None
        return client.images.get_registry_data(self.base).id

    def to_dockerfile(self) -> str:
        """Generate a Dockerfile from the layer sequence. Pure — no Docker needed."""
        return self._render_dockerfile(from_ref=self.base)

    def _to_build_dockerfile(self, *, client: Any) -> str:
        from_ref = self.base
        if self.pin_digest:
            digest = self._resolve_digest(client=client)
            if digest:
                from_ref = f"{self.base}@{digest}"
        return self._render_dockerfile(from_ref=from_ref)

    def _render_dockerfile(self, *, from_ref: str) -> str:
        lines: list[str] = ["# syntax=docker/dockerfile:1.7"]

        for stage in self.stages:
            stage_df = stage.spec._render_dockerfile(from_ref=stage.spec.base)
            stage_lines = stage_df.split("\n")
            # Drop the embedded stage's syntax directive: it must appear exactly
            # once, as the first line of the combined Dockerfile (Docker rejects
            # "only one syntax parser directive can be used").
            stage_lines = stage_lines[1:]
            if stage_lines and stage_lines[0].startswith("FROM "):
                base = validate_image_ref(stage.spec.base, field="stage base image")
                name = validate_name(stage.name, field="stage name")
                stage_lines[0] = f"FROM {base} AS {name}"
            lines.extend(stage_lines)
            lines.append("")

        lines.append(f"FROM {validate_image_ref(from_ref, field='base image')}")
        for i, layer in enumerate(self.layers):
            lines.append("")
            lines.extend(render_dockerfile(self, layer, i))
        return "\n".join(lines) + "\n"

    def _prepare_build_context(self, dockerfile: str) -> tuple[str, str]:
        """Create a build context directory from Copy layers across all stages.

        Collects Copy layers from the main spec AND all stage specs, copies their
        local sources into a temp context directory, and rewrites COPY paths in
        the Dockerfile to use context-relative paths.

        Returns ``(context_path, effective_dockerfile)``. The effective
        Dockerfile is the one backends must build: rewritten when a staged
        context was created, the input unchanged otherwise.
        """
        import shutil
        import tempfile

        all_copy_layers: list[Copy] = [
            layer for layer in self.layers if isinstance(layer, Copy) and layer.content_hash
        ]
        for stage in self.stages:
            all_copy_layers.extend(
                layer
                for layer in stage.spec.layers
                if isinstance(layer, Copy) and layer.content_hash
            )

        if not all_copy_layers:
            return ".", dockerfile

        context_dir = tempfile.mkdtemp(prefix="containerspec-context-")
        rewritten_dockerfile = dockerfile

        for layer in all_copy_layers:
            src_path = Path(layer.src)
            if not src_path.exists():
                continue
            dest_name = f"ctx_{hash(layer.src) & 0xFFFFFFFF:08x}_{src_path.name}"
            ctx_dest = Path(context_dir) / dest_name
            if src_path.is_dir():
                shutil.copytree(src_path, ctx_dest)
            else:
                shutil.copy2(src_path, ctx_dest)
            rewritten_dockerfile = rewritten_dockerfile.replace(
                f"COPY {layer.src} {layer.dest}",
                f"COPY {dest_name} {layer.dest}",
            )

        df_path = Path(context_dir) / "Dockerfile"
        df_path.write_text(rewritten_dockerfile)

        return context_dir, rewritten_dockerfile

    def resolve_chown_uid_gid(self, chown: Chown, *, index: int) -> tuple[int, int]:
        if chown.uid is not None and chown.gid is not None:
            return chown.uid, chown.gid
        if chown.uid is not None or chown.gid is not None:
            raise ValueError(
                f'chown("{chown.path}"): uid and gid must both be specified or both omitted'
            )
        for layer in reversed(self.layers[:index]):
            if isinstance(layer, User):
                return layer.uid, layer.gid
        raise ValueError(
            f'chown("{chown.path}"): no preceding .user() layer. '
            f"Add .user() before .chown(), or specify uid/gid explicitly."
        )

    async def build(
        self,
        target: str | BuildTarget,
        *,
        client: Any = None,
        backend: Any = None,
    ) -> Any:
        """Build an artifact. Uses TaskGroup for concurrent builds when called in parallel."""
        from containerspec.backends import auto_detect_backend
        from containerspec.targets import DockerTarget

        if isinstance(target, str):
            target = DockerTarget(name=target)
        if client is None and target.needs_client:
            import docker

            client = docker.from_env()
        if backend is None:
            backend = auto_detect_backend(target=target)

        hash_str = self.content_hash(client=client)
        hash_16 = hash_str[:16]

        logger.info(
            "containerspec.build.start",
            extra={"target": type(target).__name__, "hash": hash_16, "base": self.base},
        )

        if target.exists(hash=hash_16, client=client):
            logger.info("containerspec.build.cache_hit", extra={"hash": hash_16})
            return target.result_from_cache(hash=hash_16, client=client)

        logger.info("containerspec.build.cache_miss", extra={"hash": hash_16})

        tag = f"{target.name}:sha-{hash_16}"
        canonical = self._canonical_json(client=client)
        dockerfile = self._to_build_dockerfile(client=client)
        pull = not self.pin_digest

        context_path, dockerfile = self._prepare_build_context(dockerfile)

        try:
            result = await target.export(
                dockerfile=dockerfile,
                tag=tag,
                canonical_json=canonical,
                client=client,
                backend=backend,
                pull=pull,
                context_path=context_path,
            )
        except (MissingToolError, BuildError):
            raise
        except Exception as e:
            import tempfile

            fd, dockerfile_debug_path = tempfile.mkstemp(
                suffix=".Dockerfile", prefix=f"containerspec-failed-{hash_16}-"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(dockerfile)
            except OSError:
                pass
            logger.error(
                "containerspec.build.failed",
                extra={"hash": hash_16, "dockerfile": dockerfile_debug_path},
            )
            raise BuildError(
                f"Build failed for {target.name}:sha-{hash_16}. "
                f"Failed Dockerfile saved to {dockerfile_debug_path}",
            ) from e
        logger.info("containerspec.build.complete", extra={"hash": hash_16, "tag": tag})
        return result

    def _resolve_hf_home(self) -> str:
        """Resolve HF_HOME from env layers. Used by consumers that need it."""
        for layer in self.layers:
            if isinstance(layer, Env) and "HF_HOME" in layer.vars:
                return layer.vars["HF_HOME"]
        return "/root/.cache/huggingface"

    def _resolve_uid(self) -> int:
        """Resolve uid from user layers. Used by consumers that need it."""
        for layer in reversed(self.layers):
            if isinstance(layer, User):
                return layer.uid
        return 0

    def __repr__(self) -> str:
        return f"ImageSpec(base={self.base!r}, layers={len(self.layers)})"
