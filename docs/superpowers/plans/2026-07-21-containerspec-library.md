# ContainerSpec Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python library (`containerspec`) that produces content-hashed, BuildKit-cached Docker images, Firecracker rootfs ext4 images, and OCI tarballs from a fluent immutable spec — with multi-target build support, async build via `docker buildx`, `docker` as a lazy optional dependency, sidecar metadata for non-image targets, and full golden-file test coverage.

**Architecture:** Modular src layout. `layers.py` defines the `Layer` discriminated union + payload serialization. `spec.py` defines `ImageSpec` — fluent API, content hash (target-agnostic), Dockerfile generation, chown resolution, async `build()` dispatch. `targets.py` defines the `BuildTarget` protocol (`exists()`/`export()`/`result_from_cache()`) + `DockerTarget`/`FirecrackerRootfsTarget`/`OciTarget` + result types. `backends.py` defines the `BuildBackend` protocol + `BuildKitBackend` (async `docker buildx` subprocess) / `DockerBackend` (docker-py fallback). `rootfs.py` handles ext4 creation via `mke2fs -d` + sidecar metadata. `__init__.py` re-exports the public API. The content hash is on `ImageSpec` (target-agnostic) — same spec, same hash, any output. Each `BuildTarget` owns its `exists()` check so cache-skip holds uniformly.

**Tech Stack:** Python 3.12+, docker-py 7.0.0+ (optional, lazy import for Docker target), `docker buildx` CLI (async subprocess for BuildKit solve+export), `e2fsprogs`/`mke2fs` (for Firecracker rootfs), pytest + pytest-asyncio + pytest-cov, ruff, pyright (strict), testcontainers (moby/buildkit for integration tests), mkdocs-material.

**Spec:** `docs/superpowers/specs/2026-07-21-containerspec-library-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/containerspec/layers.py` (NEW) | `Layer` base + all layer dataclasses + `layer_payload()` serialization to canonical dict |
| `src/containerspec/spec.py` (NEW) | `ImageSpec` frozen dataclass — fluent API, `_canonical_payload`, `content_hash`, `tag`, `to_dockerfile`, `_resolve_chown_uid_gid`, `build()` (async), `_resolve_hf_home`, `_resolve_uid` |
| `src/containerspec/targets.py` (NEW) | `BuildTarget` protocol (`name`, `needs_client`, `exists`, `export`, `result_from_cache`), `DockerTarget`, `FirecrackerRootfsTarget`, `OciTarget`, result types (`BuiltImage`, `FirecrackerRootfs`, `OciArtifact`) |
| `src/containerspec/backends.py` (NEW) | `BuildBackend` protocol (`solve_and_export` async), `BuildKitBackend` (async `docker buildx build` via `asyncio.create_subprocess_exec`), `DockerBackend` (docker-py fallback), `MissingToolError`, `BuildError` |
| `src/containerspec/rootfs.py` (NEW) | `create_ext4()` via `mke2fs -d`, `write_sidecar()` / `read_sidecar()` (atomic temp+rename), `MissingToolError` check |
| `src/containerspec/__init__.py` (MODIFY) | Re-export all public API from submodules |
| `src/containerspec/py.typed` (EXISTS) | PEP 561 marker |
| `tests/conftest.py` (MODIFY) | Shared fixtures: `mock_docker_client`, `mock_buildx_backend`, `tmp_rootfs_path` |
| `tests/test_api.py` (NEW) | Fluent API, immutability, layer types, method chaining |
| `tests/test_hash.py` (NEW) | Hash contract, golden payload, golden hash, lazy digest, uid=0 vs null, apt vs apk |
| `tests/test_dockerfile.py` (NEW) | Golden-file Dockerfile rendering (apt rootless, apk rootless, root) |
| `tests/test_chown.py` (NEW) | Chown resolution (all cases + edge cases) |
| `tests/test_targets.py` (NEW) | BuildTarget protocol, exists() per target, export() invocation, result_from_cache() |
| `tests/test_backends.py` (NEW) | Backend selection, BuildKitBackend CLI invocation (mocked async subprocess), DockerBackend fallback |
| `tests/test_rootfs.py` (NEW) | ext4 creation (mke2fs -d), sidecar write/read (atomic), MissingToolError |
| `tests/test_build.py` (NEW) | build() async with mocked backend: existence-check skip, export, label/sidecar, result types |
| `tests/test_public_api.py` (NEW) | to_dockerfile(), content_hash(), tag(), __repr__ |
| `tests/golden/rootless_apt_dockerfile.txt` (NEW) | Golden fixture |
| `tests/golden/rootless_apk_dockerfile.txt` (NEW) | Golden fixture |
| `tests/golden/root_dockerfile.txt` (NEW) | Golden fixture |
| `tests/test_integration.py` (NEW) | Real BuildKit via testcontainers moby/buildkit, gated behind CONTAINERSPEC_E2E=1 |

---

## Task 1: Layer types and payload serialization

**Files:**
- Create: `src/containerspec/layers.py`
- Modify: `src/containerspec/__init__.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
from __future__ import annotations

from containerspec import (
    AddPython,
    ApkInstall,
    AptInstall,
    Chown,
    Env,
    Entrypoint,
    ImageSpec,
    Layer,
    PipInstall,
    RunCommands,
    UvPipInstall,
    User,
    Workdir,
)


class TestImageSpecFluentApi:
    def test_from_registry_creates_empty_spec(self) -> None:
        spec = ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04")
        assert spec.base == "nvidia/cuda:13.3.0-devel-ubuntu24.04"
        assert spec.layers == ()

    def test_pin_digest_default_true(self) -> None:
        assert ImageSpec.from_registry("base").pin_digest is True

    def test_pin_digest_false(self) -> None:
        assert ImageSpec.from_registry("base", pin_digest=False).pin_digest is False

    def test_add_python(self) -> None:
        spec = ImageSpec.from_registry("base").add_python("3.12")
        assert isinstance(spec.layers[0], AddPython)
        assert spec.layers[0].version == "3.12"

    def test_apt_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").apt_install("git", "build-essential")
        assert isinstance(spec.layers[0], AptInstall)
        assert spec.layers[0].packages == ("build-essential", "git")

    def test_apk_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("alpine:3.20").apk_install("openrc", "util-linux")
        assert isinstance(spec.layers[0], ApkInstall)
        assert spec.layers[0].packages == ("openrc", "util-linux")

    def test_uv_pip_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").uv_pip_install("vllm", "flashinfer==0.2.0")
        assert isinstance(spec.layers[0], UvPipInstall)
        assert spec.layers[0].packages == ("flashinfer==0.2.0", "vllm")

    def test_pip_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").pip_install("httpx", "rich")
        assert isinstance(spec.layers[0], PipInstall)
        assert spec.layers[0].packages == ("httpx", "rich")

    def test_env_sorted_by_key(self) -> None:
        spec = ImageSpec.from_registry("base").env({"B": "2", "A": "1"})
        assert isinstance(spec.layers[0], Env)
        assert dict(spec.layers[0].vars) == {"A": "1", "B": "2"}

    def test_run_commands_preserves_order(self) -> None:
        spec = ImageSpec.from_registry("base").run_commands("echo a", "echo b")
        assert isinstance(spec.layers[0], RunCommands)
        assert spec.layers[0].commands == ("echo a", "echo b")

    def test_workdir(self) -> None:
        spec = ImageSpec.from_registry("base").workdir("/app")
        assert isinstance(spec.layers[0], Workdir)
        assert spec.layers[0].path == "/app"

    def test_chown_default(self) -> None:
        spec = ImageSpec.from_registry("base").chown("/foo")
        assert isinstance(spec.layers[0], Chown)
        assert spec.layers[0].uid is None
        assert spec.layers[0].gid is None

    def test_chown_explicit(self) -> None:
        spec = ImageSpec.from_registry("base").chown("/foo", uid=0, gid=0)
        assert spec.layers[0].uid == 0
        assert spec.layers[0].gid == 0

    def test_user(self) -> None:
        spec = ImageSpec.from_registry("base").user(uid=1000, gid=1000, name="warden")
        assert isinstance(spec.layers[0], User)
        assert spec.layers[0].uid == 1000

    def test_entrypoint_empty(self) -> None:
        spec = ImageSpec.from_registry("base").entrypoint([])
        assert isinstance(spec.layers[0], Entrypoint)
        assert spec.layers[0].commands == ()

    def test_entrypoint_none(self) -> None:
        spec = ImageSpec.from_registry("base").entrypoint(None)
        assert spec.layers[0].commands is None

    def test_immutability(self) -> None:
        spec = ImageSpec.from_registry("base")
        new = spec.add_python("3.12")
        assert spec.layers == ()
        assert len(new.layers) == 1

    def test_chaining(self) -> None:
        spec = (
            ImageSpec.from_registry("base")
            .add_python("3.12")
            .uv_pip_install("vllm")
            .workdir("/app")
            .user(uid=1000, gid=1000, name="warden")
            .entrypoint([])
        )
        assert len(spec.layers) == 5
        assert [type(l).__name__ for l in spec.layers] == [
            "AddPython", "UvPipInstall", "Workdir", "User", "Entrypoint",
        ]

    def test_repr(self) -> None:
        spec = ImageSpec.from_registry("base").add_python("3.12")
        assert "base" in repr(spec)
        assert "1" in repr(spec)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'ImageSpec'`

- [ ] **Step 3: Implement `layers.py`**

Create `src/containerspec/layers.py`:

```python
"""Layer types for ImageSpec — frozen dataclasses, discriminated union."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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


@dataclass(frozen=True)
class Entrypoint(Layer):
    commands: tuple[str, ...] | None


def layer_payload(layer: Layer) -> dict[str, Any]:
    """Serialize a layer to its canonical dict for hashing."""
    if isinstance(layer, AddPython):
        return {"type": "add_python", "version": layer.version}
    if isinstance(layer, AptInstall):
        return {"type": "apt_install", "packages": list(layer.packages)}
    if isinstance(layer, ApkInstall):
        return {"type": "apk_install", "packages": list(layer.packages)}
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
        return {"type": "user", "uid": layer.uid, "gid": layer.gid, "name": layer.name}
    if isinstance(layer, Entrypoint):
        return {
            "type": "entrypoint",
            "commands": list(layer.commands) if layer.commands is not None else None,
        }
    raise TypeError(f"Unknown layer type: {type(layer).__name__}")
```

- [ ] **Step 4: Implement `spec.py` (fluent API only — no hash/dockerfile/build yet)**

Create `src/containerspec/spec.py`:

```python
"""ImageSpec — fluent, immutable Docker image specification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from containerspec.layers import (
    AddPython,
    ApkInstall,
    AptInstall,
    Chown,
    Entrypoint,
    Env,
    Layer,
    PipInstall,
    RunCommands,
    UvPipInstall,
    User,
    Workdir,
)


@dataclass(frozen=True)
class ImageSpec:
    """Fluent, immutable Docker image specification.

    Build with ``ImageSpec.from_registry(base).apt_install(...).uv_pip_install(...)``
    and call ``.build(name)`` to produce a tagged Docker image.
    """

    base: str
    pin_digest: bool = True
    layers: tuple[Layer, ...] = ()

    @classmethod
    def from_registry(cls, base: str, *, pin_digest: bool = True) -> ImageSpec:
        return cls(base=base, pin_digest=pin_digest, layers=())

    def _with(self, layer: Layer) -> ImageSpec:
        return ImageSpec(base=self.base, pin_digest=self.pin_digest, layers=(*self.layers, layer))

    def add_python(self, version: str) -> ImageSpec:
        return self._with(AddPython(version=version))

    def apt_install(self, *packages: str) -> ImageSpec:
        return self._with(AptInstall(packages=tuple(sorted(packages))))

    def apk_install(self, *packages: str) -> ImageSpec:
        return self._with(ApkInstall(packages=tuple(sorted(packages))))

    def uv_pip_install(self, *packages: str) -> ImageSpec:
        return self._with(UvPipInstall(packages=tuple(sorted(packages))))

    def pip_install(self, *packages: str) -> ImageSpec:
        return self._with(PipInstall(packages=tuple(sorted(packages))))

    def env(self, vars: Mapping[str, str]) -> ImageSpec:
        sorted_vars = {k: vars[k] for k in sorted(vars)}
        return self._with(Env(vars=sorted_vars))

    def run_commands(self, *commands: str) -> ImageSpec:
        return self._with(RunCommands(commands=tuple(commands)))

    def workdir(self, path: str) -> ImageSpec:
        return self._with(Workdir(path=path))

    def chown(self, path: str, *, uid: int | None = None, gid: int | None = None) -> ImageSpec:
        return self._with(Chown(path=path, uid=uid, gid=gid))

    def user(self, *, uid: int, gid: int, name: str) -> ImageSpec:
        return self._with(User(uid=uid, gid=gid, name=name))

    def entrypoint(self, commands: Sequence[str] | None) -> ImageSpec:
        normalized: tuple[str, ...] | None = tuple(commands) if commands is not None else None
        return self._with(Entrypoint(commands=normalized))

    def __repr__(self) -> str:
        return f"ImageSpec(base={self.base!r}, layers={len(self.layers)})"
```

- [ ] **Step 5: Update `__init__.py` to re-export**

Replace `src/containerspec/__init__.py`:

```python
"""ContainerSpec — fluent, content-hashed Docker image builder.

Mirrors Modal's Image API surface but targets any Docker daemon or BuildKit
instance via ``docker buildx``. Every method returns a new frozen ``ImageSpec``.
``docker`` is a lazy import inside ``build()`` — ``to_dockerfile()``,
``content_hash()``, and ``tag()`` work without Docker installed.
"""

from __future__ import annotations

from containerspec.layers import (
    AddPython,
    ApkInstall,
    AptInstall,
    Chown,
    Entrypoint,
    Env,
    Layer,
    PipInstall,
    RunCommands,
    UvPipInstall,
    User,
    Workdir,
)
from containerspec.spec import ImageSpec

__all__ = [
    "AddPython",
    "ApkInstall",
    "AptInstall",
    "Chown",
    "Entrypoint",
    "Env",
    "ImageSpec",
    "Layer",
    "PipInstall",
    "RunCommands",
    "UvPipInstall",
    "User",
    "Workdir",
]

__version__ = "0.1.0"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (18 tests)

- [ ] **Step 7: Lint and typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 8: Commit**

```bash
git add src/containerspec/layers.py src/containerspec/spec.py src/containerspec/__init__.py tests/test_api.py
git commit -m "feat: add ImageSpec fluent API with apt/apk/uv/pip layer types"
```

---

## Task 2: Content hash — canonical payload and hash computation

**Files:**
- Modify: `src/containerspec/spec.py`
- Create: `tests/test_hash.py`

- [ ] **Step 1: Write the failing hash tests**

Create `tests/test_hash.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock

from containerspec import ImageSpec


class TestImageSpecHash:
    def test_same_spec_same_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        b = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_reordered_apt_packages_same_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).apt_install("git", "build-essential")
        b = ImageSpec.from_registry("base", pin_digest=False).apt_install("build-essential", "git")
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_reordered_layers_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).user(uid=1000, gid=1000, name="w").chown("/foo")
        b = ImageSpec.from_registry("base", pin_digest=False).chown("/foo", uid=1000, gid=1000).user(uid=1000, gid=1000, name="w")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_uv_vs_pip_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        b = ImageSpec.from_registry("base", pin_digest=False).pip_install("vllm")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_apt_vs_apk_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).apt_install("pkg")
        b = ImageSpec.from_registry("base", pin_digest=False).apk_install("pkg")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_different_base_different_hash(self) -> None:
        a = ImageSpec.from_registry("base-a", pin_digest=False)
        b = ImageSpec.from_registry("base-b", pin_digest=False)
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_pin_digest_false_no_client_needed(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False)
        h = spec.content_hash(client=None)
        assert len(h) == 64

    def test_pin_digest_false_skips_resolution(self) -> None:
        client = MagicMock()
        spec = ImageSpec.from_registry("base", pin_digest=False)
        spec.content_hash(client=client)
        client.images.get_registry_data.assert_not_called()

    def test_pin_digest_resolved_digest_feeds_hash(self) -> None:
        client = MagicMock()
        client.images.get_registry_data.side_effect = [
            MagicMock(id="sha256:aaa111"),
            MagicMock(id="sha256:bbb222"),
        ]
        a = ImageSpec.from_registry("base", pin_digest=True)
        b = ImageSpec.from_registry("base", pin_digest=True)
        assert a.content_hash(client=client) != b.content_hash(client=client)

    def test_from_registry_does_not_resolve_until_hash(self) -> None:
        client = MagicMock()
        _ = ImageSpec.from_registry("base", pin_digest=True)
        client.images.get_registry_data.assert_not_called()

    def test_chown_default_uid_null_in_payload(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).user(uid=1000, gid=1000, name="w").chown("/foo")
        b = ImageSpec.from_registry("base", pin_digest=False).user(uid=9999, gid=9999, name="x").chown("/foo")
        a_payload = a._canonical_payload(client=None)
        b_payload = b._canonical_payload(client=None)
        a_chown = [l for l in a_payload["layers"] if l["type"] == "chown"][0]
        b_chown = [l for l in b_payload["layers"] if l["type"] == "chown"][0]
        assert a_chown["uid"] is None
        assert b_chown["uid"] is None

    def test_chown_explicit_uid_zero_in_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/foo", uid=0, gid=0)
        payload = spec._canonical_payload(client=None)
        chown = [l for l in payload["layers"] if l["type"] == "chown"][0]
        assert chown["uid"] == 0
        assert chown["gid"] == 0

    def test_user_uid_feeds_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).user(uid=1000, gid=1000, name="w")
        b = ImageSpec.from_registry("base", pin_digest=False).user(uid=1001, gid=1000, name="w")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_tag_format(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        tag = spec.tag("warden/vllm", client=None)
        assert tag.startswith("warden/vllm:sha-")
        assert len(tag.split("sha-")[1]) == 16

    def test_tag_with_pin_digest_needs_client(self) -> None:
        import pytest
        spec = ImageSpec.from_registry("base", pin_digest=True)
        with pytest.raises(ValueError, match="pin_digest=True requires a docker client"):
            spec.tag("warden/vllm", client=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hash.py -v`
Expected: FAIL with `AttributeError: 'ImageSpec' object has no attribute 'content_hash'`

- [ ] **Step 3: Implement hash methods**

Add to `ImageSpec` in `src/containerspec/spec.py` (after `entrypoint`, before `__repr__`):

```python
    def _canonical_payload(self, *, client: Any) -> dict[str, Any]:
        """Build the canonical hash payload — the full readable layer list."""
        if self.pin_digest:
            if client is None:
                raise ValueError(
                    "pin_digest=True requires a docker client to resolve the base image digest. "
                    "Either pass client= to content_hash()/tag()/build(), or use pin_digest=False."
                )
            digest = client.images.get_registry_data(self.base).id
            base_entry: dict[str, Any] = {"ref": self.base, "digest": digest}
        else:
            base_entry = {"ref": self.base}
        return {
            "base": base_entry,
            "pin_digest": self.pin_digest,
            "layers": [layer_payload(layer) for layer in self.layers],
        }

    def _canonical_json(self, *, client: Any) -> str:
        import json
        return json.dumps(self._canonical_payload(client=client), sort_keys=True)

    def content_hash(self, *, client: Any = None) -> str:
        """Full sha256 hex of the canonical payload. Target-agnostic."""
        import hashlib
        return hashlib.sha256(self._canonical_json(client=client).encode()).hexdigest()

    def tag(self, name: str, *, client: Any = None) -> str:
        """Return ``{name}:sha-{hash[:16]}`` without building."""
        return f"{name}:sha-{self.content_hash(client=client)[:16]}"
```

Add `Any` to the imports: `from typing import Any, Mapping, Sequence`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hash.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/containerspec/spec.py tests/test_hash.py
git commit -m "feat: add content hash with canonical payload (target-agnostic, apt/apk discriminator)"
```

---

## Task 3: Golden hash tests (load-bearing — cache contract lock)

**Files:**
- Modify: `tests/test_hash.py`

- [ ] **Step 1: Write the golden canonical payload test**

Add to `tests/test_hash.py`:

```python
import pytest


class TestGoldenHash:
    """Locks the cache contract. See spec for full rationale."""

    _GOLDEN_SPEC = (
        ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
        .add_python("3.12")
        .apt_install("git", "build-essential")
        .uv_pip_install("vllm", "flashinfer==0.2.0")
        .env({"HF_HOME": "/home/warden/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .workdir("/app")
        .user(uid=1000, gid=1000, name="warden")
        .chown("/home/warden/.cache/huggingface")
        .entrypoint([])
    )

    _GOLDEN_CANONICAL_JSON = (
        '{"base": {"ref": "nvidia/cuda:13.3.0-devel-ubuntu24.04"}, '
        '"layers": ['
        '{"type": "add_python", "version": "3.12"}, '
        '{"type": "apt_install", "packages": ["build-essential", "git"]}, '
        '{"type": "uv_pip_install", "packages": ["flashinfer==0.2.0", "vllm"]}, '
        '{"type": "env", "vars": {"HF_HOME": "/home/warden/.cache/huggingface", '
        '"HF_HUB_ENABLE_HF_TRANSFER": "1"}}, '
        '{"type": "workdir", "path": "/app"}, '
        '{"type": "user", "uid": 1000, "gid": 1000, "name": "warden"}, '
        '{"type": "chown", "path": "/home/warden/.cache/huggingface", "uid": null, "gid": null}, '
        '{"type": "entrypoint", "commands": []}'
        '], "pin_digest": false}'
    )

    _GOLDEN_HASH = "REPLACE_ME_WITH_ACTUAL_HASH"

    def test_golden_canonical_payload(self) -> None:
        actual = self._GOLDEN_SPEC._canonical_json(client=None)
        assert actual == self._GOLDEN_CANONICAL_JSON, (
            f"Canonical payload drift.\nExpected: {self._GOLDEN_CANONICAL_JSON}\n"
            f"Actual:   {actual}"
        )

    def test_golden_hash_stable(self) -> None:
        if self._GOLDEN_HASH == "REPLACE_ME_WITH_ACTUAL_HASH":
            actual = self._GOLDEN_SPEC.content_hash(client=None)[:16]
            pytest.fail(f"Replace _GOLDEN_HASH with: {actual}")
        actual = self._GOLDEN_SPEC.content_hash(client=None)[:16]
        assert actual == self._GOLDEN_HASH, f"Hash drift. Expected: {self._GOLDEN_HASH}, Actual: {actual}"
```

- [ ] **Step 2: Run the canonical payload test, fix JSON literal if needed**

Run: `uv run pytest tests/test_hash.py::TestGoldenHash::test_golden_canonical_payload -v`
Expected: PASS — or FAIL showing expected vs actual. If it fails, eyeball the actual JSON, confirm it's correct, and update `_GOLDEN_CANONICAL_JSON`.

- [ ] **Step 3: Get the actual hash and fill in `_GOLDEN_HASH`**

Run: `uv run pytest tests/test_hash.py::TestGoldenHash::test_golden_hash_stable -v`
Expected: FAIL with `"Replace _GOLDEN_HASH with: <actual hash>"`. Copy the hash into `_GOLDEN_HASH`.

- [ ] **Step 4: Re-run both golden tests**

Run: `uv run pytest tests/test_hash.py::TestGoldenHash -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_hash.py
git commit -m "test: golden canonical payload and hash stability (cache contract lock)"
```

---

## Task 4: Dockerfile generation

**Files:**
- Modify: `src/containerspec/spec.py`
- Create: `tests/test_dockerfile.py`
- Create: `tests/golden/rootless_apt_dockerfile.txt`
- Create: `tests/golden/rootless_apk_dockerfile.txt`
- Create: `tests/golden/root_dockerfile.txt`

- [ ] **Step 1: Write the failing Dockerfile tests**

Create `tests/test_dockerfile.py`:

```python
from __future__ import annotations

from pathlib import Path

from containerspec import ImageSpec


class TestDockerfile:
    def test_rootless_apt_dockerfile(self) -> None:
        spec = (
            ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
            .add_python("3.12")
            .apt_install("git", "build-essential")
            .uv_pip_install("vllm", "flashinfer==0.2.0")
            .pip_install("huggingface_hub[hf_transfer]")
            .env({"HF_HOME": "/home/warden/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
            .workdir("/app")
            .user(uid=1000, gid=1000, name="warden")
            .chown("/home/warden/.cache/huggingface")
            .entrypoint([])
        )
        actual = spec.to_dockerfile()
        expected = (Path(__file__).parent / "golden" / "rootless_apt_dockerfile.txt").read_text()
        assert actual == expected, f"Expected:\n{expected}\nActual:\n{actual}"

    def test_rootless_apk_dockerfile(self) -> None:
        spec = (
            ImageSpec.from_registry("alpine:3.20", pin_digest=False)
            .apk_install("openrc", "util-linux")
            .run_commands("ln -s agetty /etc/init.d/agetty.ttyS0")
            .env({"TERM": "linux"})
            .user(uid=1000, gid=1000, name="warden")
            .entrypoint([])
        )
        actual = spec.to_dockerfile()
        expected = (Path(__file__).parent / "golden" / "rootless_apk_dockerfile.txt").read_text()
        assert actual == expected, f"Expected:\n{expected}\nActual:\n{actual}"

    def test_root_dockerfile(self) -> None:
        spec = (
            ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
            .add_python("3.12")
            .uv_pip_install("vllm")
            .env({"HF_HOME": "/root/.cache/huggingface"})
            .entrypoint([])
        )
        actual = spec.to_dockerfile()
        expected = (Path(__file__).parent / "golden" / "root_dockerfile.txt").read_text()
        assert actual == expected, f"Expected:\n{expected}\nActual:\n{actual}"

    def test_add_python_lines(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).add_python("3.12")
        df = spec.to_dockerfile()
        assert "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/" in df
        assert "uv python install 3.12" in df
        assert "uv venv --python 3.12 /opt/venv" in df
        assert "ENV PATH=/opt/venv/bin:$PATH" in df

    def test_no_python_when_not_added(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        df = spec.to_dockerfile()
        assert "uv venv" not in df
        assert "PATH=/opt/venv" not in df

    def test_workdir(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).workdir("/app")
        df = spec.to_dockerfile()
        assert "WORKDIR /app" in df

    def test_entrypoint_clear(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).entrypoint([])
        df = spec.to_dockerfile()
        assert "ENTRYPOINT []" in df

    def test_entrypoint_none_omits(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).entrypoint(None)
        df = spec.to_dockerfile()
        assert "ENTRYPOINT" not in df
```

- [ ] **Step 2: Create golden fixtures**

Create `tests/golden/rootless_apt_dockerfile.txt`:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

# add_python("3.12")
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# apt_install("build-essential", "git")
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential git

# uv_pip_install("flashinfer==0.2.0", "vllm")
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_LINK_MODE=copy uv pip install --system flashinfer==0.2.0 vllm

# pip_install("huggingface_hub[hf_transfer]")
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir huggingface_hub[hf_transfer]

# env({"HF_HOME": "/home/warden/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
ENV HF_HOME=/home/warden/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1

# workdir("/app")
WORKDIR /app

# chown("/home/warden/.cache/huggingface") — resolved to uid=1000, gid=1000 from preceding .user()
RUN mkdir -p /home/warden/.cache/huggingface && chown -R 1000:1000 /home/warden/.cache/huggingface

# user(uid=1000, gid=1000, name="warden")
RUN groupadd -g 1000 warden && useradd -u 1000 -g 1000 -m -d /home/warden warden
USER 1000:1000

# entrypoint([])
ENTRYPOINT []
```

Create `tests/golden/rootless_apk_dockerfile.txt`:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM alpine:3.20

# apk_install("openrc", "util-linux")
RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \
    apk add --no-cache openrc util-linux

# run_commands("ln -s agetty /etc/init.d/agetty.ttyS0")
RUN ln -s agetty /etc/init.d/agetty.ttyS0

# env({"TERM": "linux"})
ENV TERM=linux

# user(uid=1000, gid=1000, name="warden")
RUN addgroup -g 1000 warden && adduser -u 1000 -G warden -D -h /home/warden warden
USER 1000:1000

# entrypoint([])
ENTRYPOINT []
```

Create `tests/golden/root_dockerfile.txt`:

```dockerfile
# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

# add_python("3.12")
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# uv_pip_install("vllm")
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_LINK_MODE=copy uv pip install --system vllm

# env({"HF_HOME": "/root/.cache/huggingface"})
ENV HF_HOME=/root/.cache/huggingface

# entrypoint([])
ENTRYPOINT []
```

- [ ] **Step 3: Implement `to_dockerfile()` and `_resolve_chown_uid_gid()`**

Add to `ImageSpec` in `src/containerspec/spec.py` (after `tag`, before `__repr__`):

```python
    def to_dockerfile(self) -> str:
        """Generate a Dockerfile from the layer sequence. Pure — no Docker needed."""
        lines: list[str] = ["# syntax=docker/dockerfile:1.7", f"FROM {self.base}"]

        for i, layer in enumerate(self.layers):
            if i > 0:
                lines.append("")

            if isinstance(layer, AddPython):
                lines.append(f'# add_python("{layer.version}")')
                lines.append("COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/")
                lines.append(
                    f"RUN uv python install {layer.version} && "
                    f"uv venv --python {layer.version} /opt/venv"
                )
                lines.append("ENV PATH=/opt/venv/bin:$PATH")

            elif isinstance(layer, AptInstall):
                pkgs = " ".join(layer.packages)
                lines.append(f"# apt_install({', '.join(repr(p) for p in layer.packages)})")
                lines.append(
                    "RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \\\n"
                    "    --mount=type=cache,target=/var/lib/apt,sharing=locked \\\n"
                    f"    apt-get update && apt-get install -y --no-install-recommends {pkgs}"
                )

            elif isinstance(layer, ApkInstall):
                pkgs = " ".join(layer.packages)
                lines.append(f"# apk_install({', '.join(repr(p) for p in layer.packages)})")
                lines.append(
                    "RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \\\n"
                    f"    apk add --no-cache {pkgs}"
                )

            elif isinstance(layer, UvPipInstall):
                pkgs = " ".join(layer.packages)
                lines.append(f"# uv_pip_install({', '.join(repr(p) for p in layer.packages)})")
                lines.append(
                    "RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \\\n"
                    f"    UV_LINK_MODE=copy uv pip install --system {pkgs}"
                )

            elif isinstance(layer, PipInstall):
                pkgs = " ".join(layer.packages)
                lines.append(f"# pip_install({', '.join(repr(p) for p in layer.packages)})")
                lines.append(
                    "RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \\\n"
                    f"    pip install --no-cache-dir {pkgs}"
                )

            elif isinstance(layer, Env):
                env_str = " ".join(f"{k}={v}" for k, v in layer.vars.items())
                lines.append(f"# env({dict(layer.vars)})")
                lines.append(f"ENV {env_str}")

            elif isinstance(layer, RunCommands):
                lines.append(f"# run_commands({', '.join(repr(c) for c in layer.commands)})")
                for cmd in layer.commands:
                    lines.append(f"RUN {cmd}")

            elif isinstance(layer, Workdir):
                lines.append(f'# workdir("{layer.path}")')
                lines.append(f"WORKDIR {layer.path}")

            elif isinstance(layer, Chown):
                uid, gid = self._resolve_chown_uid_gid(layer, index=i)
                source = " from preceding .user()" if layer.uid is None and layer.gid is None else ""
                lines.append(f'# chown("{layer.path}") — resolved to uid={uid}, gid={gid}{source}')
                lines.append(f"RUN mkdir -p {layer.path} && chown -R {uid}:{gid} {layer.path}")

            elif isinstance(layer, User):
                lines.append(f'# user(uid={layer.uid}, gid={layer.gid}, name="{layer.name}")')
                lines.append(
                    f"RUN groupadd -g {layer.gid} {layer.name} && "
                    f"useradd -u {layer.uid} -g {layer.gid} -m -d /home/{layer.name} {layer.name}"
                )
                lines.append(f"USER {layer.uid}:{layer.gid}")

            elif isinstance(layer, Entrypoint):
                if layer.commands is None:
                    lines.append("# entrypoint(None)")
                else:
                    lines.append(f"# entrypoint([{', '.join(repr(c) for c in layer.commands)}])")
                    lines.append(f"ENTRYPOINT {list(layer.commands)}")

        return "\n".join(lines) + "\n"

    def _resolve_chown_uid_gid(self, chown: Chown, *, index: int) -> tuple[int, int]:
        """Resolve chown uid/gid at render time."""
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
```

- [ ] **Step 4: Run tests, fix golden files if needed**

Run: `uv run pytest tests/test_dockerfile.py -v`
Expected: PASS (8 tests). If golden files don't match, compare expected vs actual and update the golden file if the actual output is correct.

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/containerspec/spec.py tests/test_dockerfile.py tests/golden/
git commit -m "feat: add Dockerfile generation with apt/apk + golden-file tests"
```

---

## Task 5: Chown resolution tests

**Files:**
- Create: `tests/test_chown.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_chown.py`:

```python
from __future__ import annotations

import pytest

from containerspec import ImageSpec


class TestChownResolution:
    def test_no_user_no_explicit_raises(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/foo")
        with pytest.raises(ValueError, match="no preceding .user"):
            spec.to_dockerfile()

    def test_default_resolves_to_preceding_user(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="warden")
            .chown("/home/warden/cache")
        )
        assert "chown -R 1000:1000 /home/warden/cache" in spec.to_dockerfile()

    def test_explicit_override_without_user(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/foo", uid=0, gid=0)
        assert "chown -R 0:0 /foo" in spec.to_dockerfile()

    def test_partial_uid_gid_raises(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/foo", uid=1000, gid=None)
        with pytest.raises(ValueError, match="both be specified"):
            spec.to_dockerfile()

    def test_resolves_to_most_recent_user(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="warden")
            .user(uid=2000, gid=2000, name="other")
            .chown("/foo")
        )
        assert "chown -R 2000:2000 /foo" in spec.to_dockerfile()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_chown.py -v`
Expected: PASS (5 tests — implementation already exists from Task 4)

- [ ] **Step 3: Commit**

```bash
git add tests/test_chown.py
git commit -m "test: add chown resolution tests (default, explicit, error cases)"
```

---

## Task 6: Build targets — `BuildTarget` protocol + concrete targets + result types

**Files:**
- Create: `src/containerspec/targets.py`
- Modify: `src/containerspec/__init__.py`
- Create: `tests/test_targets.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_targets.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

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
        assert target.exists(hash="abc123", client=None) is False

    def test_exists_true_when_sidecar_hash_matches(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        rootfs_path.write_bytes(b"fake ext4")
        sidecar = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar.write_text(json.dumps({"hash": "sha-abc123def4567890", "spec": {}}))
        target = FirecrackerRootfsTarget(path=str(rootfs_path))
        assert target.exists(hash="abc123def4567890", client=None) is True

    def test_exists_false_when_sidecar_hash_mismatch(self, tmp_path: Path) -> None:
        rootfs_path = tmp_path / "rootfs.ext4"
        rootfs_path.write_bytes(b"fake ext4")
        sidecar = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar.write_text(json.dumps({"hash": "sha-different", "spec": {}}))
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


class TestOciTarget:
    def test_needs_client_false(self) -> None:
        assert OciTarget(path="/tmp/tar.tar").needs_client is False

    def test_exists_false_when_no_sidecar(self, tmp_path: Path) -> None:
        target = OciTarget(path=str(tmp_path / "tar.tar"))
        assert target.exists(hash="abc123", client=None) is False

    def test_result_from_cache(self, tmp_path: Path) -> None:
        oci_path = tmp_path / "tar.tar"
        oci_path.write_bytes(b"fake tar")
        target = OciTarget(path=str(oci_path))
        result = target.result_from_cache(hash="abc123def4567890", client=None)
        assert isinstance(result, OciArtifact)
        assert result.path == str(oci_path)
        assert result.hash == "abc123def4567890"


class TestResultTypes:
    def test_built_image_fields(self) -> None:
        r = BuiltImage(tag="x:sha-abc", hf_home="/root/.cache/huggingface", uid=0)
        assert r.tag == "x:sha-abc"
        assert r.hf_home == "/root/.cache/huggingface"
        assert r.uid == 0

    def test_firecracker_rootfs_fields(self) -> None:
        r = FirecrackerRootfs(path="/tmp/rootfs.ext4", hash="abc", size_mb=2048)
        assert r.path == "/tmp/rootfs.ext4"
        assert r.hash == "abc"
        assert r.size_mb == 2048

    def test_oci_artifact_fields(self) -> None:
        r = OciArtifact(path="/tmp/tar.tar", hash="abc")
        assert r.path == "/tmp/tar.tar"
        assert r.hash == "abc"
```

- [ ] **Step 2: Implement `targets.py`**

Create `src/containerspec/targets.py`:

```python
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
    hf_home: str
    uid: int


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
        self, *, dockerfile: str, tag: str, canonical_json: str,
        client: Any | None, backend: Any,
    ) -> Any: ...

    def result_from_cache(self, *, hash: str, client: Any | None) -> Any: ...


def _sidecar_path(artifact_path: str) -> Path:
    return Path(f"{artifact_path}.containerspec.json")


def _sidecar_exists(artifact_path: str, hash_str: str) -> bool:
    sidecar = _sidecar_path(artifact_path)
    if not sidecar.exists():
        return False
    try:
        data = json.loads(sidecar.read_text())
        return data.get("hash") == f"sha-{hash_str}"
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
        self, *, dockerfile: str, tag: str, canonical_json: str,
        client: Any | None, backend: Any,
    ) -> BuiltImage:
        await backend.solve_and_export(
            dockerfile=dockerfile, tag=tag, output_type="docker",
            output_path=None, labels={"containerspec.image_spec": canonical_json}, pull=True,
        )
        return BuiltImage(tag=tag, hf_home="", uid=0)

    def result_from_cache(self, *, hash: str, client: Any | None) -> BuiltImage:
        tag = f"{self.name}:sha-{hash}"
        return BuiltImage(tag=tag, hf_home="", uid=0)


@dataclass(frozen=True)
class FirecrackerRootfsTarget:
    """Build a Firecracker rootfs ext4 image. No Docker daemon needed."""
    path: str
    size_mb: int = 1024

    @property
    def name(self) -> str:
        return "containerspec-rootfs"

    @property
    def needs_client(self) -> bool:
        return False

    def exists(self, *, hash: str, client: Any | None) -> bool:
        return _sidecar_exists(self.path, hash)

    async def export(
        self, *, dockerfile: str, tag: str, canonical_json: str,
        client: Any | None, backend: Any,
    ) -> FirecrackerRootfs:
        from containerspec.rootfs import create_ext4, write_sidecar, check_mke2fs
        import tempfile

        check_mke2fs()
        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.solve_and_export(
                dockerfile=dockerfile, tag=tag, output_type="local",
                output_path=tmpdir, labels={}, pull=True,
            )
            create_ext4(rootfs_dir=tmpdir, dest=self.path, size_mb=self.size_mb)
        write_sidecar(f"{self.path}.containerspec.json", hash_str=tag, spec_json=canonical_json)
        return FirecrackerRootfs(path=self.path, hash=tag, size_mb=self.size_mb)

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
        self, *, dockerfile: str, tag: str, canonical_json: str,
        client: Any | None, backend: Any,
    ) -> OciArtifact:
        await backend.solve_and_export(
            dockerfile=dockerfile, tag=tag, output_type="oci",
            output_path=self.path, labels={}, pull=True,
        )
        from containerspec.rootfs import write_sidecar
        write_sidecar(f"{self.path}.containerspec.json", hash_str=tag, spec_json=canonical_json)
        return OciArtifact(path=self.path, hash=tag)

    def result_from_cache(self, *, hash: str, client: Any | None) -> OciArtifact:
        return OciArtifact(path=self.path, hash=hash)
```

- [ ] **Step 3: Update `__init__.py` to re-export targets**

Add to `src/containerspec/__init__.py`:

```python
from containerspec.targets import (
    BuiltImage,
    BuildTarget,
    DockerTarget,
    FirecrackerRootfs,
    FirecrackerRootfsTarget,
    OciArtifact,
    OciTarget,
)
```

And add to `__all__`:
```python
    "BuiltImage",
    "BuildTarget",
    "DockerTarget",
    "FirecrackerRootfs",
    "FirecrackerRootfsTarget",
    "OciArtifact",
    "OciTarget",
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_targets.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/containerspec/targets.py src/containerspec/__init__.py tests/test_targets.py
git commit -m "feat: add BuildTarget protocol with Docker/Firecracker/OCI targets"
```

---

## Task 7: Rootfs creation — ext4 via `mke2fs -d`, sidecar metadata

**Files:**
- Create: `src/containerspec/rootfs.py`
- Create: `tests/test_rootfs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rootfs.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from containerspec.rootfs import (
    MissingToolError,
    check_mke2fs,
    create_ext4,
    read_sidecar,
    write_sidecar,
)


class TestCheckMke2fs:
    def test_raises_when_mke2fs_not_found(self) -> None:
        with patch("containerspec.rootfs.shutil.which", return_value=None):
            with pytest.raises(MissingToolError, match="mke2fs not found"):
                check_mke2fs()

    def test_passes_when_mke2fs_found(self) -> None:
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/sbin/mke2fs"):
            check_mke2fs()


class TestCreateExt4:
    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_calls_mke2fs_with_correct_args(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.wait.return_value = 0
        mock_exec.return_value = proc
        with patch("containerspec.rootfs.check_mke2fs"):
            await create_ext4(rootfs_dir="/tmp/rootfs", dest="/tmp/rootfs.ext4", size_mb=512)
        cmd = mock_exec.call_args.args
        assert "mke2fs" in cmd[0]
        assert "-t" in cmd
        assert "ext4" in cmd
        assert "-d" in cmd
        assert "/tmp/rootfs" in cmd
        assert "/tmp/rootfs.ext4" in cmd

    @patch("containerspec.rootfs.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_raises_on_nonzero_exit(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.wait.return_value = 1
        mock_exec.return_value = proc
        with patch("containerspec.rootfs.check_mke2fs"):
            with pytest.raises(RuntimeError, match="mke2fs failed"):
                await create_ext4(rootfs_dir="/tmp/rootfs", dest="/tmp/rootfs.ext4", size_mb=512)


class TestWriteSidecar:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "rootfs.ext4.containerspec.json"
        write_sidecar(str(sidecar_path), hash_str="sha-abc123", spec_json='{"base": {}}')
        data = json.loads(sidecar_path.read_text())
        assert data["hash"] == "sha-abc123"
        assert data["spec"] == {"base": {}}

    def test_atomic_write(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "rootfs.ext4.containerspec.json"
        write_sidecar(str(sidecar_path), hash_str="sha-abc", spec_json='{}')
        assert sidecar_path.exists()
        assert not (tmp_path / "rootfs.ext4.containerspec.json.tmp").exists()


class TestReadSidecar:
    def test_reads_valid_sidecar(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar_path.write_text(json.dumps({"hash": "sha-abc123", "spec": {"base": {}}}))
        data = read_sidecar(str(sidecar_path))
        assert data["hash"] == "sha-abc123"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert read_sidecar(str(tmp_path / "nonexistent.json")) is None

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        sidecar_path = tmp_path / "bad.json"
        sidecar_path.write_text("not json")
        assert read_sidecar(str(sidecar_path)) is None
```

- [ ] **Step 2: Implement `rootfs.py`**

Create `src/containerspec/rootfs.py`:

```python
"""Rootfs creation — ext4 via mke2fs -d, sidecar metadata."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any


class MissingToolError(RuntimeError):
    """Raised when a required host tool (e.g. mke2fs) is not found."""


def check_mke2fs() -> None:
    """Raise MissingToolError if mke2fs is not on PATH."""
    if shutil.which("mke2fs") is None:
        raise MissingToolError(
            "mke2fs not found — install e2fsprogs to build Firecracker rootfs images"
        )


async def create_ext4(*, rootfs_dir: str, dest: str, size_mb: int) -> None:
    """Create an ext4 image at dest, populated from rootfs_dir. No root needed."""
    check_mke2fs()
    size_bytes = size_mb * 1024 * 1024
    proc = await asyncio.create_subprocess_exec(
        "mke2fs", "-t", "ext4", "-d", rootfs_dir, dest, str(size_bytes),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"mke2fs failed with exit code {rc}")


def write_sidecar(path: str, *, hash_str: str, spec_json: str) -> None:
    """Write sidecar metadata atomically (temp file + rename)."""
    data = {"hash": hash_str, "spec": json.loads(spec_json)}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, sort_keys=True)
    os.rename(tmp_path, path)


def read_sidecar(path: str) -> dict[str, Any] | None:
    """Read sidecar metadata. Returns None if missing or invalid."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
```

- [ ] **Step 3: Add `pytest-asyncio` config if needed**

Ensure `pyproject.toml` has:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_rootfs.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/containerspec/rootfs.py tests/test_rootfs.py pyproject.toml
git commit -m "feat: add rootfs creation (mke2fs -d) and sidecar metadata"
```

---

## Task 8: Build backends — `BuildBackend` protocol + `BuildKitBackend` + `DockerBackend`

**Files:**
- Create: `src/containerspec/backends.py`
- Modify: `src/containerspec/__init__.py`
- Create: `tests/test_backends.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backends.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from containerspec import BuildKitBackend, DockerBackend


class TestBuildKitBackend:
    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_solve_and_export_calls_buildx(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.wait.return_value = 0
        mock_exec.return_value = proc
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n", tag="x:sha-abc",
            output_type="docker", output_path=None,
            labels={"containerspec.image_spec": "{}"}, pull=True,
        )
        cmd = mock_exec.call_args.args
        assert "docker" in cmd[0]
        assert "buildx" in cmd
        assert "build" in cmd
        assert "--output" in cmd

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_oci_output_type(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.wait.return_value = 0
        mock_exec.return_value = proc
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n", tag="x:sha-abc",
            output_type="oci", output_path="/tmp/tar.tar",
            labels={}, pull=True,
        )
        cmd = mock_exec.call_args.args
        output_idx = cmd.index("--output")
        assert "type=oci" in cmd[output_idx + 1]
        assert "/tmp/tar.tar" in cmd[output_idx + 1]

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_local_output_type(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.wait.return_value = 0
        mock_exec.return_value = proc
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n", tag="x:sha-abc",
            output_type="local", output_path="/tmp/rootfs",
            labels={}, pull=True,
        )
        cmd = mock_exec.call_args.args
        output_idx = cmd.index("--output")
        assert "type=local" in cmd[output_idx + 1]
        assert "/tmp/rootfs" in cmd[output_idx + 1]

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_raises_on_nonzero_exit(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.wait.return_value = 1
        mock_exec.return_value = proc
        backend = BuildKitBackend()
        with pytest.raises(RuntimeError, match="buildx build failed"):
            await backend.solve_and_export(
                dockerfile="FROM base\n", tag="x:sha-abc",
                output_type="docker", output_path=None,
                labels={}, pull=True,
            )

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_dockerfile_via_stdin(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.wait.return_value = 0
        mock_exec.return_value = proc
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n", tag="x:sha-abc",
            output_type="docker", output_path=None,
            labels={}, pull=True,
        )
        assert mock_exec.call_args.kwargs["stdin"] is not None


class TestDockerBackend:
    @pytest.mark.asyncio
    async def test_build_calls_images_build(self) -> None:
        client = MagicMock()
        client.images.build.return_value = (MagicMock(), [])
        backend = DockerBackend(client=client)
        await backend.solve_and_export(
            dockerfile="FROM base\n", tag="x:sha-abc",
            output_type="docker", output_path=None,
            labels={"containerspec.image_spec": "{}"}, pull=True,
        )
        client.images.build.assert_called_once()
        kwargs = client.images.build.call_args.kwargs
        assert kwargs["tag"] == "x:sha-abc"
        assert kwargs["pull"] is True
        assert kwargs["rm"] is True


class TestAutoDetect:
    @patch("containerspec.backends.shutil.which")
    def test_buildx_available(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/bin/docker"
        from containerspec.backends import auto_detect_backend
        backend = auto_detect_backend()
        assert isinstance(backend, BuildKitBackend)

    @patch("containerspec.backends.shutil.which")
    def test_buildx_unavailable_fallback(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        from containerspec.backends import auto_detect_backend
        backend = auto_detect_backend()
        assert isinstance(backend, DockerBackend)
```

- [ ] **Step 2: Implement `backends.py`**

Create `src/containerspec/backends.py`:

```python
"""Build backends — BuildKitBackend (docker buildx) and DockerBackend (docker-py)."""

from __future__ import annotations

import asyncio
import io
import shutil
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BuildBackend(Protocol):
    """A build execution backend."""

    async def solve_and_export(
        self, *, dockerfile: str, tag: str, output_type: str,
        output_path: str | None, labels: dict[str, str], pull: bool,
    ) -> None: ...


@dataclass(frozen=True)
class BuildKitBackend:
    """Build via `docker buildx build` CLI subprocess. Supports all targets."""
    url: str | None = None
    builder: str | None = None

    async def solve_and_export(
        self, *, dockerfile: str, tag: str, output_type: str,
        output_path: str | None, labels: dict[str, str], pull: bool,
    ) -> None:
        cmd: list[str] = ["docker", "buildx", "build"]
        if self.builder:
            cmd.extend(["--builder", self.builder])
        output_val = output_type
        if output_path:
            output_val = f"type={output_type},dest={output_path}"
        cmd.extend(["--output", output_val, "--tag", tag])
        if pull:
            cmd.append("--pull")
        for k, v in labels.items():
            cmd.extend(["--label", f"{k}={v}"])
        cmd.append("-")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        proc.stdin.write(dockerfile.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(f"buildx build failed with exit code {rc}")


@dataclass
class DockerBackend:
    """Build via docker-py. Docker target only (fallback when buildx unavailable)."""
    client: Any = None

    async def solve_and_export(
        self, *, dockerfile: str, tag: str, output_type: str,
        output_path: str | None, labels: dict[str, str], pull: bool,
    ) -> None:
        if self.client is None:
            import docker
            self.client = docker.from_env()
        self.client.images.build(
            fileobj=io.BytesIO(dockerfile.encode()),
            tag=tag,
            pull=pull,
            rm=True,
            labels=labels,
        )


def auto_detect_backend() -> BuildBackend:
    """Pick BuildKitBackend if docker buildx is available, else DockerBackend."""
    if shutil.which("docker") is not None:
        return BuildKitBackend()
    return DockerBackend()
```

- [ ] **Step 3: Update `__init__.py` to re-export backends**

Add to `src/containerspec/__init__.py`:

```python
from containerspec.backends import (
    BuildBackend,
    BuildKitBackend,
    DockerBackend,
)
```

And add to `__all__`:
```python
    "BuildBackend",
    "BuildKitBackend",
    "DockerBackend",
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_backends.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Lint and typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/containerspec/backends.py src/containerspec/__init__.py tests/test_backends.py
git commit -m "feat: add BuildBackend protocol with BuildKit/Docker backends (async)"
```

---

## Task 9: `ImageSpec.build()` — async, target dispatch, existence check, result

**Files:**
- Modify: `src/containerspec/spec.py`
- Modify: `src/containerspec/__init__.py`
- Create: `tests/test_build.py`

- [ ] **Step 1: Write the failing build tests**

Create `tests/test_build.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from containerspec import (
    BuiltImage,
    DockerTarget,
    FirecrackerRootfs,
    FirecrackerRootfsTarget,
    ImageSpec,
    OciArtifact,
    OciTarget,
)


class TestBuildDockerTarget:
    @pytest.mark.asyncio
    async def test_skips_when_image_exists(self) -> None:
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        result = await spec.build("warden/test", client=client)
        assert isinstance(result, BuiltImage)
        client.images.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_builds_when_not_found(self) -> None:
        client = MagicMock()
        client.images.get.side_effect = Exception("not found")
        backend = AsyncMock()
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        result = await spec.build("warden/test", client=client, backend=backend)
        assert isinstance(result, BuiltImage)
        backend.solve_and_export.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_hf_home_from_env(self) -> None:
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        spec = ImageSpec.from_registry("base", pin_digest=False).env({"HF_HOME": "/custom"}).uv_pip_install("vllm")
        result = await spec.build("warden/test", client=client)
        assert result.hf_home == "/custom"

    @pytest.mark.asyncio
    async def test_returns_default_hf_home(self) -> None:
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        result = await spec.build("warden/test", client=client)
        assert result.hf_home == "/root/.cache/huggingface"

    @pytest.mark.asyncio
    async def test_returns_uid_from_user(self) -> None:
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        spec = ImageSpec.from_registry("base", pin_digest=False).user(uid=1000, gid=1000, name="w").uv_pip_install("vllm")
        result = await spec.build("warden/test", client=client)
        assert result.uid == 1000

    @pytest.mark.asyncio
    async def test_returns_uid_zero_without_user(self) -> None:
        client = MagicMock()
        client.images.get.return_value = MagicMock()
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        result = await spec.build("warden/test", client=client)
        assert result.uid == 0


class TestBuildFirecrackerRootfs:
    @pytest.mark.asyncio
    async def test_dispatches_to_backend(self, tmp_path) -> None:
        backend = AsyncMock()
        spec = ImageSpec.from_registry("alpine:3.20", pin_digest=False).apk_install("openrc")
        target = FirecrackerRootfsTarget(path=str(tmp_path / "rootfs.ext4"), size_mb=512)
        with patch("containerspec.rootfs.check_mke2fs"):
            with patch("containerspec.rootfs.create_ext4", new_callable=AsyncMock):
                result = await spec.build(target, backend=backend)
        assert isinstance(result, FirecrackerRootfs)
        backend.solve_and_export.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_sidecar_exists(self, tmp_path) -> None:
        import json
        rootfs_path = tmp_path / "rootfs.ext4"
        rootfs_path.write_bytes(b"fake")
        spec = ImageSpec.from_registry("alpine:3.20", pin_digest=False).apk_install("openrc")
        hash_str = spec.content_hash(client=None)[:16]
        sidecar = tmp_path / "rootfs.ext4.containerspec.json"
        sidecar.write_text(json.dumps({"hash": f"sha-{hash_str}", "spec": {}}))
        target = FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=512)
        backend = AsyncMock()
        result = await spec.build(target, backend=backend)
        assert isinstance(result, FirecrackerRootfs)
        backend.solve_and_export.assert_not_called()


class TestBuildOciTarget:
    @pytest.mark.asyncio
    async def test_dispatches_to_backend(self, tmp_path) -> None:
        backend = AsyncMock()
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        target = OciTarget(path=str(tmp_path / "tar.tar"))
        result = await spec.build(target, backend=backend)
        assert isinstance(result, OciArtifact)
        backend.solve_and_export.assert_called_once()
```

- [ ] **Step 2: Implement `build()` + `_resolve_hf_home()` + `_resolve_uid()`**

Add to `ImageSpec` in `src/containerspec/spec.py` (after `to_dockerfile`, before `__repr__`):

```python
    async def build(
        self, target: str | BuildTarget, *, client: Any = None, backend: Any = None,
    ) -> Any:
        """Build an artifact (Docker image, rootfs, or OCI tarball).

        If the artifact already exists (``target.exists()``), skips the build.
        Requires ``docker buildx`` CLI for non-Docker targets, ``docker`` Python
        package for Docker target existence check.
        """
        if isinstance(target, str):
            target = DockerTarget(name=target)
        if client is None and target.needs_client:
            import docker
            client = docker.from_env()
        if backend is None:
            from containerspec.backends import auto_detect_backend
            backend = auto_detect_backend()

        hash_str = self.content_hash(client=client)
        tag = f"{target.name}:sha-{hash_str[:16]}"

        if target.exists(hash=hash_str[:16], client=client):
            result = target.result_from_cache(hash=hash_str[:16], client=client)
            return self._enrich_result(result)

        canonical = self._canonical_json(client=client)
        result = await target.export(
            dockerfile=self.to_dockerfile(), tag=tag, canonical_json=canonical,
            client=client, backend=backend,
        )
        return self._enrich_result(result)

    def _enrich_result(self, result: Any) -> Any:
        """Fill in hf_home/uid for BuiltImage from the spec."""
        from containerspec import BuiltImage
        if isinstance(result, BuiltImage) and result.hf_home == "":
            return BuiltImage(
                tag=result.tag,
                hf_home=self._resolve_hf_home(),
                uid=self._resolve_uid(),
            )
        return result

    def _resolve_hf_home(self) -> str:
        for layer in self.layers:
            if isinstance(layer, Env) and "HF_HOME" in layer.vars:
                return layer.vars["HF_HOME"]
        return "/root/.cache/huggingface"

    def _resolve_uid(self) -> int:
        for layer in reversed(self.layers):
            if isinstance(layer, User):
                return layer.uid
        return 0
```

Add imports to `spec.py`:
```python
from containerspec.targets import BuildTarget, DockerTarget
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_build.py -v`
Expected: PASS (all tests)

- [ ] **Step 4: Lint and typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 5: Commit**

```bash
git add src/containerspec/spec.py src/containerspec/__init__.py tests/test_build.py
git commit -m "feat: add async ImageSpec.build() with multi-target dispatch"
```

---

## Task 10: Public API tests

**Files:**
- Create: `tests/test_public_api.py`

- [ ] **Step 1: Write tests**

Create `tests/test_public_api.py`:

```python
from __future__ import annotations

from containerspec import ImageSpec


class TestPublicApi:
    def test_to_dockerfile_no_docker(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        df = spec.to_dockerfile()
        assert isinstance(df, str)
        assert "FROM base" in df

    def test_content_hash_no_docker(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        h = spec.content_hash(client=None)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_tag_no_docker(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        tag = spec.tag("warden/vllm", client=None)
        assert tag.startswith("warden/vllm:sha-")
        assert len(tag.split("sha-")[1]) == 16

    def test_repr_shows_base_and_layer_count(self) -> None:
        spec = ImageSpec.from_registry("base").add_python("3.12").uv_pip_install("vllm")
        repr_str = repr(spec)
        assert "base" in repr_str
        assert "2" in repr_str

    def test_all_exports_available(self) -> None:
        import containerspec
        for name in [
            "ImageSpec", "Layer", "AddPython", "AptInstall", "ApkInstall",
            "UvPipInstall", "PipInstall", "Env", "RunCommands", "Workdir",
            "Chown", "User", "Entrypoint", "BuiltImage", "FirecrackerRootfs",
            "OciArtifact", "BuildTarget", "DockerTarget",
            "FirecrackerRootfsTarget", "OciTarget", "BuildBackend",
            "BuildKitBackend", "DockerBackend",
        ]:
            assert hasattr(containerspec, name), f"Missing export: {name}"
```

- [ ] **Step 2: Run, commit**

Run: `uv run pytest tests/test_public_api.py -v`
Expected: PASS

```bash
git add tests/test_public_api.py
git commit -m "test: public API surface (to_dockerfile, content_hash, tag, repr, exports)"
```

---

## Task 11: Integration test (gated behind CONTAINERSPEC_E2E=1)

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write gated integration tests**

Create `tests/test_integration.py`:

```python
"""Real BuildKit integration tests. Gated behind CONTAINERSPEC_E2E=1."""

from __future__ import annotations

import os
import subprocess

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _require_docker() -> None:
    if os.environ.get("CONTAINERSPEC_E2E") != "1":
        pytest.skip("set CONTAINERSPEC_E2E=1 to run real-BuildKit tests")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:  # noqa: S603, S607
        pytest.skip("docker daemon not reachable")


@pytest.mark.asyncio
async def test_docker_target_build_skip_run() -> None:
    from containerspec import ImageSpec

    spec = (
        ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
        .pip_install("httpx")
        .entrypoint([])
    )
    built = await spec.build("containerspec-test")
    assert built.tag.startswith("containerspec-test:sha-")

    built_again = await spec.build("containerspec-test")
    assert built_again.tag == built.tag

    result = subprocess.run(  # noqa: S603
        ["docker", "run", "--rm", built.tag, "python", "-c", "import httpx; print('ok')"],  # noqa: S607
        capture_output=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert b"ok" in result.stdout

    subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607


@pytest.mark.asyncio
async def test_oci_target_build(tmp_path) -> None:
    from containerspec import OciTarget, ImageSpec

    spec = (
        ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
        .pip_install("httpx")
        .entrypoint([])
    )
    oci_path = tmp_path / "image.tar"
    result = await spec.build(OciTarget(path=str(oci_path)))
    assert oci_path.exists()
    assert result.path == str(oci_path)


@pytest.mark.asyncio
async def test_firecracker_rootfs_target_build(tmp_path) -> None:
    from containerspec import FirecrackerRootfsTarget, ImageSpec

    spec = (
        ImageSpec.from_registry("alpine:3.20", pin_digest=False)
        .apk_install("openrc")
        .entrypoint([])
    )
    rootfs_path = tmp_path / "rootfs.ext4"
    result = await spec.build(FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=256))
    assert rootfs_path.exists()
    assert result.path == str(rootfs_path)
    assert (tmp_path / "rootfs.ext4.containerspec.json").exists()
```

- [ ] **Step 2: Run if Docker available**

Run: `CONTAINERSPEC_E2E=1 uv run pytest tests/test_integration.py -v -s`
Expected: PASS, or skip if no Docker.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: real-BuildKit integration tests (gated behind CONTAINERSPEC_E2E=1)"
```

---

## Task 12: Full verification + docs update

- [ ] **Step 1: Run all unit tests with coverage**

Run: `uv run pytest tests/ -v --cov=containerspec --cov-report=term-missing`
Expected: All pass, coverage >= 95%

- [ ] **Step 2: Lint + typecheck**

Run: `uv run ruff check --fix src/ tests/ && uv run pyright src/`
Expected: 0 errors

- [ ] **Step 3: Verify mkdocs builds**

Run: `uv run mkdocs build --strict`
Expected: PASS

- [ ] **Step 4: Update docs**

Update `docs/api-reference/containerspec.md` to reference the full API. Update `docs/getting-started/quick-start.md` with multi-target examples. Update `README.md` with Firecracker/OCI examples and runtime requirements section.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: full verification + docs update for v0.1"
```

---

## Self-Review Notes

- **Spec coverage:** Layer types incl. apk (Task 1), hash with apt/apk discriminator + golden tests (Tasks 2-3), Dockerfile generation for apt/apk (Task 4), chown resolution (Task 5), BuildTarget protocol with exists()/export()/result_from_cache() (Task 6), rootfs creation via mke2fs -d + sidecar (Task 7), BuildBackend async with BuildKit/Docker (Task 8), async build() with multi-target dispatch (Task 9), public API (Task 10), integration tests (Task 11), verification (Task 12).
- **No placeholders:** `_GOLDEN_HASH` is intentional — filled from first run.
- **Type consistency:** `BuildTarget` protocol in `targets.py`, `BuildBackend` in `backends.py`, both re-exported. `build()` is async, `solve_and_export()` is async, `export()` is async. `DockerTarget.needs_client=True`, rootfs/OCI `needs_client=False`.
- **v0.1 ships all three targets** per user's decision. All targets are async, use BuildKit via buildx subprocess, and have per-target existence checks (sidecar for non-image targets).
