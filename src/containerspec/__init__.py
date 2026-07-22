"""ContainerSpec — fluent, content-hashed image builder for Docker, Firecracker, and OCI.

Mirrors Modal's Image API surface but targets any Docker daemon, BuildKit instance,
or Buildah (daemonless) for producing Docker images, Firecracker rootfs ext4 images,
and OCI tarballs. Every method returns a new frozen ``ImageSpec``.

Convenience methods (``nvm_install``, ``npm_install``, ``brew_install``, ``rust_install``,
``cargo_install``, ``uvx_install``) generate the correct bootstrap + install + PATH setup.
For custom tooling, use ``run_commands()`` directly.

``docker`` is a lazy import inside ``build()`` — ``to_dockerfile()``,
``content_hash()``, and ``tag()`` work without Docker installed.
"""

from __future__ import annotations

from containerspec.backends import (
    BuildahBackend,
    BuildBackend,
    BuildError,
    BuildKitBackend,
    DockerBackend,
)
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
)
from containerspec.rootfs import MissingToolError
from containerspec.spec import ImageSpec, StageSpec
from containerspec.targets import (
    BuildTarget,
    BuiltImage,
    DockerTarget,
    FirecrackerRootfs,
    FirecrackerRootfsTarget,
    OciArtifact,
    OciTarget,
)

__all__ = [
    "AddPython",
    "ApkInstall",
    "AptInstall",
    "BrewInstall",
    "BuildBackend",
    "BuildError",
    "BuildKitBackend",
    "BuildTarget",
    "BuildahBackend",
    "BuiltImage",
    "CargoInstall",
    "Chown",
    "Cmd",
    "Copy",
    "CopyFromStage",
    "DnfInstall",
    "DockerBackend",
    "DockerTarget",
    "Entrypoint",
    "Env",
    "Expose",
    "FirecrackerRootfs",
    "FirecrackerRootfsTarget",
    "GemInstall",
    "GoInstall",
    "ImageSpec",
    "Layer",
    "MissingToolError",
    "NpmInstall",
    "NvmInstall",
    "OciArtifact",
    "OciTarget",
    "PacmanInstall",
    "PipInstall",
    "PnpmInstall",
    "RunCommands",
    "RustInstall",
    "StageSpec",
    "User",
    "UvPipInstall",
    "UvxInstall",
    "Volume",
    "Workdir",
    "YarnInstall",
    "ZypperInstall",
]

__version__ = "0.1.4"
