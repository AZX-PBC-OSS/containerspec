"""Layer types for ImageSpec — frozen dataclasses, discriminated union."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Layer:
    """Base type for all image layers."""


@dataclass(frozen=True)
class AddPython(Layer):
    version: str


@dataclass(frozen=True)
class AptInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class ApkInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class DnfInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class PacmanInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class ZypperInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class UvPipInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class PipInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class Env(Layer):
    vars: Mapping[str, str]


@dataclass(frozen=True)
class RunCommands(Layer):
    commands: tuple[str, ...]


@dataclass(frozen=True)
class Workdir(Layer):
    path: str


@dataclass(frozen=True)
class Chown(Layer):
    path: str
    uid: int | None = None
    gid: int | None = None


@dataclass(frozen=True)
class User(Layer):
    uid: int
    gid: int
    name: str
    alpine: bool = False


@dataclass(frozen=True)
class Entrypoint(Layer):
    commands: tuple[str, ...] | None


@dataclass(frozen=True)
class Expose(Layer):
    ports: tuple[int, ...]


@dataclass(frozen=True)
class Cmd(Layer):
    commands: tuple[str, ...] | None


@dataclass(frozen=True)
class Volume(Layer):
    paths: tuple[str, ...]


@dataclass(frozen=True)
class Copy(Layer):
    src: str
    dest: str
    content_hash: str = ""


@dataclass(frozen=True)
class NvmInstall(Layer):
    version: str


@dataclass(frozen=True)
class NpmInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class PnpmInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class BrewInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class RustInstall(Layer):
    pass


@dataclass(frozen=True)
class CargoInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class UvxInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class YarnInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class GemInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class GoInstall(Layer):
    packages: tuple[str, ...]


@dataclass(frozen=True)
class CopyFromStage(Layer):
    stage_name: str
    stage_hash: str
    src: str
    dest: str


def frozen_mapping(vars: Mapping[str, str]) -> Mapping[str, str]:
    """Wrap a dict in MappingProxyType for immutability."""
    return MappingProxyType(dict(vars))


def layer_payload(layer: Layer) -> dict[str, Any]:
    """Serialize a layer to its canonical dict for hashing."""
    if isinstance(layer, AddPython):
        return {"type": "add_python", "version": layer.version}
    if isinstance(layer, AptInstall):
        return {"type": "apt_install", "packages": list(layer.packages)}
    if isinstance(layer, ApkInstall):
        return {"type": "apk_install", "packages": list(layer.packages)}
    if isinstance(layer, DnfInstall):
        return {"type": "dnf_install", "packages": list(layer.packages)}
    if isinstance(layer, PacmanInstall):
        return {"type": "pacman_install", "packages": list(layer.packages)}
    if isinstance(layer, ZypperInstall):
        return {"type": "zypper_install", "packages": list(layer.packages)}
    if isinstance(layer, UvPipInstall):
        return {"type": "uv_pip_install", "packages": list(layer.packages)}
    if isinstance(layer, PipInstall):
        return {"type": "pip_install", "packages": list(layer.packages)}
    if isinstance(layer, Env):
        return {"type": "env", "vars": dict(layer.vars)}
    if isinstance(layer, RunCommands):
        return {"type": "run_commands", "commands": list(layer.commands)}
    if isinstance(layer, Workdir):
        return {"type": "workdir", "path": layer.path}
    if isinstance(layer, Chown):
        return {"type": "chown", "path": layer.path, "uid": layer.uid, "gid": layer.gid}
    if isinstance(layer, User):
        return {
            "type": "user",
            "uid": layer.uid,
            "gid": layer.gid,
            "name": layer.name,
            "alpine": layer.alpine,
        }
    if isinstance(layer, Entrypoint):
        return {
            "type": "entrypoint",
            "commands": list(layer.commands) if layer.commands is not None else None,
        }
    if isinstance(layer, Expose):
        return {"type": "expose", "ports": list(layer.ports)}
    if isinstance(layer, Cmd):
        return {
            "type": "cmd",
            "commands": list(layer.commands) if layer.commands is not None else None,
        }
    if isinstance(layer, Volume):
        return {"type": "volume", "paths": list(layer.paths)}
    if isinstance(layer, Copy):
        return {
            "type": "copy",
            "src": layer.src,
            "dest": layer.dest,
            "content_hash": layer.content_hash,
        }
    if isinstance(layer, NvmInstall):
        return {"type": "nvm_install", "version": layer.version}
    if isinstance(layer, NpmInstall):
        return {"type": "npm_install", "packages": list(layer.packages)}
    if isinstance(layer, PnpmInstall):
        return {"type": "pnpm_install", "packages": list(layer.packages)}
    if isinstance(layer, BrewInstall):
        return {"type": "brew_install", "packages": list(layer.packages)}
    if isinstance(layer, RustInstall):
        return {"type": "rust_install"}
    if isinstance(layer, CargoInstall):
        return {"type": "cargo_install", "packages": list(layer.packages)}
    if isinstance(layer, UvxInstall):
        return {"type": "uvx_install", "packages": list(layer.packages)}
    if isinstance(layer, YarnInstall):
        return {"type": "yarn_install", "packages": list(layer.packages)}
    if isinstance(layer, GemInstall):
        return {"type": "gem_install", "packages": list(layer.packages)}
    if isinstance(layer, GoInstall):
        return {"type": "go_install", "packages": list(layer.packages)}
    if isinstance(layer, CopyFromStage):
        return {
            "type": "copy_from_stage",
            "stage_name": layer.stage_name,
            "stage_hash": layer.stage_hash,
            "src": layer.src,
            "dest": layer.dest,
        }
    raise TypeError(f"Unknown layer type: {type(layer).__name__}")
