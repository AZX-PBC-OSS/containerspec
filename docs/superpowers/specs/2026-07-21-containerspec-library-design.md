# ContainerSpec Library Design

**Date:** 2026-07-21
**Status:** Approved (design phase — implementation plan to follow)
**Scope:** Standalone Python library (`containerspec`) for building container images and VM root filesystems from a fluent, immutable, content-hashed spec. BuildKit-native with multi-target output (Docker image, Firecracker rootfs, OCI artifact). Pure spec operations work without Docker installed. Consumed by Warden via PyPI.

## Goal

A reusable Python library that lets you build Docker images **and Firecracker VM root filesystems** dynamically from a fluent spec — naming base image, Python version, apt/apk/pip/uv packages with version pins, env vars, rootless user, and arbitrary commands — with content-hash-based caching so the same spec never rebuilds from scratch, regardless of output target.

The API mirrors Modal's `Image` builder ergonomics but targets any BuildKit-capable backend (local Docker daemon, remote `buildkitd`, or `moby/buildkit` container) instead of Modal's cloud. No Kubernetes, no orchestration — just "declare packages, get a tagged image or a bootable rootfs, skip if it already exists."

## Why a standalone library

1. **The pure builder has no domain knowledge.** `ImageSpec` doesn't know about LLMs, vLLM, SGLang, HuggingFace, or Warden's `DeploySpec`. It knows about package managers, content hashes, and BuildKit export targets. Keeping it in Warden couples an unrelated concern to a specific application.
2. **Two consumers with different output targets.** Warden's `LocalBackend` wants Docker images for local GPU inference. Warden's sandbox Firecracker runtime wants ext4 rootfs images for VMs. Both consume the same `ImageSpec` — only the `BuildTarget` differs. A standalone library with a clean target protocol serves both.
3. **The API is self-contained.** No Warden imports, no `DeploySpec`, no `VLLMTask`. The library depends only on `docker` (docker-py, for daemon-side existence checks and the Docker target's image-load) and the `docker buildx` CLI (for BuildKit solve+export). `to_dockerfile()` and `content_hash()` work without Docker or buildx installed.
4. **Open-sourcing is near-free.** MIT license, `py.typed` marker, a README. The maintenance cost (BuildKit/buildx API churn) is already paid by Warden using it in its GPU e2e suite. If external adoption shows up, that's found value; if it doesn't, nothing was lost.

## Package layout

```
containerspec/
├── pyproject.toml              # standalone package, MIT, py>=3.12
├── README.md                   # quick start, API reference, examples
├── LICENSE                     # MIT
├── src/
│   └── containerspec/
│       ├── __init__.py         # public API (ImageSpec, Layer types, BuiltImage, targets)
│       ├── layers.py           # Layer dataclasses + payload serialization
│       ├── spec.py             # ImageSpec — fluent API, hash, dockerfile render
│       ├── targets.py          # BuildTarget protocol + DockerTarget, FirecrackerRootfsTarget, OciTarget
│       ├── backends.py         # BuildBackend protocol + BuildKitBackend, DockerBackend
│       ├── rootfs.py           # ext4 rootfs creation (mke2fs -d, sidecar metadata)
│       └── py.typed            # PEP 561 marker
├── tests/
│   ├── conftest.py             # shared fixtures (mock buildx, mock docker client, tmp rootfs)
│   ├── test_api.py             # fluent API, immutability, layer types
│   ├── test_hash.py            # hash contract, golden payload, golden hash, lazy digest
│   ├── test_dockerfile.py      # golden-file Dockerfile rendering (apt + apk)
│   ├── test_chown.py           # chown resolution (all cases)
│   ├── test_targets.py         # BuildTarget protocol, exists(), export() per target
│   ├── test_backends.py        # backend selection, BuildKit CLI invocation
│   ├── test_rootfs.py          # ext4 creation, sidecar metadata, mke2fs -d
│   ├── test_build.py           # build() with mocked backend, existence check, label/sidecar
│   ├── test_public_api.py      # to_dockerfile(), content_hash(), tag(), __repr__
│   ├── golden/
│   │   ├── rootless_apt_dockerfile.txt
│   │   ├── rootless_apk_dockerfile.txt
│   │   └── root_dockerfile.txt
│   └── test_integration.py     # real BuildKit via moby/buildkit, gated behind CONTAINERSPEC_E2E=1
└── docs/
```

## Public API

```python
from containerspec import (
    ImageSpec,
    DockerTarget,
    FirecrackerRootfsTarget,
    OciTarget,
    BuiltImage,
    FirecrackerRootfs,
    OciArtifact,
)

# Docker image build (common path — string shortcut):
image = (
    ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04")
    .add_python("3.12")
    .apt_install("git", "build-essential", "ffmpeg")
    .uv_pip_install("vllm", "flashinfer==0.2.0")
    .env({"HF_HOME": "/home/warden/.cache/huggingface"})
    .user(uid=1000, gid=1000, name="warden")
    .chown("/home/warden/.cache/huggingface")
    .entrypoint([])
)
built: BuiltImage = image.build("warden/vllm")

# Firecracker rootfs (same spec, different target — no Docker daemon needed):
rootfs: FirecrackerRootfs = image.build(
    FirecrackerRootfsTarget(path="/path/rootfs.ext4", size_mb=2048)
)

# OCI tarball (same spec, different target):
oci: OciArtifact = image.build(OciTarget(path="/path/tar.tar"))

# Pure operations — no Docker or buildx needed:
dockerfile: str = image.to_dockerfile()
hash_str: str = image.content_hash(client=None)   # pin_digest=False required for client=None
tag: str = image.tag("warden/vllm")               # "warden/vllm:sha-abc123def4567890"
```

### `ImageSpec`

Frozen dataclass. Every method returns a new `ImageSpec` (immutable + fluent, Modal parity).

| Method | Returns | Notes |
|---|---|---|
| `from_registry(base, *, pin_digest=True)` | `ImageSpec` | Class method. `pin_digest=True` resolves base manifest digest for hash (lazy — at `content_hash`/`build` time, not construction). |
| `add_python(version)` | `ImageSpec` | uv-managed Python at `/opt/venv`. |
| `apt_install(*packages)` | `ImageSpec` | Sorted within layer. Ubuntu/Debian base images. |
| `apk_install(*packages)` | `ImageSpec` | Sorted within layer. Alpine base images (Firecracker rootfs use case). |
| `uv_pip_install(*packages)` | `ImageSpec` | Sorted within layer. Uses `uv pip install --system`. |
| `pip_install(*packages)` | `ImageSpec` | Sorted within layer. Fallback for packages uv can't resolve. |
| `env(vars: Mapping[str, str])` | `ImageSpec` | Sorted by key for hash stability. |
| `run_commands(*commands)` | `ImageSpec` | Order preserved (commands can depend on each other). |
| `workdir(path)` | `ImageSpec` | Sets `WORKDIR`. |
| `chown(path, *, uid=None, gid=None)` | `ImageSpec` | Default uid/gid: `None` → resolve from preceding `.user()` at render time. Explicit values override. |
| `user(*, uid, gid, name)` | `ImageSpec` | `groupadd`/`useradd`/`USER`. Pure primitive — no chown, no domain knowledge. |
| `entrypoint(commands)` | `ImageSpec` | `None` = keep base image's. `[]` = clear. |
| `to_dockerfile()` | `str` | Pure — no Docker/buildx needed. |
| `content_hash(*, client=None)` | `str` | Full sha256 hex. Target-agnostic. `client` required when `pin_digest=True`. `None` client works when `pin_digest=False`. |
| `tag(name, *, client=None)` | `str` | `"{name}:sha-{hash[:16]}"`. `client` required when `pin_digest=True`. Pure when `pin_digest=False` (`client=None`). |
| `build(target, *, client=None, backend=None)` | `BuiltImage \| FirecrackerRootfs \| OciArtifact` | **Async.** `target` is `str` (Docker shortcut) or `BuildTarget` instance. Lazy-imports `docker`/invokes `buildx` via `asyncio.create_subprocess_exec`. Checks `target.exists(hash)` first (skip if built). Uses `@overload` for precise return types: `build(str) -> BuiltImage`, `build(FirecrackerRootfsTarget) -> FirecrackerRootfs`, `build(OciTarget) -> OciArtifact`. Concurrent builds via `asyncio.gather(spec1.build(...), spec2.build(...))` — BuildKit cache handles parallel solves safely. |

### `BuildTarget` protocol

The extension point for output formats. Each target owns its existence-check (the cache-skip property holds uniformly, not just for Docker) and its export logic.

```python
@runtime_checkable
class BuildTarget(Protocol):
    @property
    def name(self) -> str: ...
        # Tag prefix for BuildKit: --tag {name}:sha-{hash[:16]}.
        # DockerTarget: the image name ("warden/vllm"). RootfsTarget/OciTarget: an
        # internal BuildKit reference (the real artifact goes to `path`).

    @property
    def needs_client(self) -> bool: ...
        # True when the target needs a docker-py client (existence check on daemon).
        # DockerTarget: True. FirecrackerRootfsTarget/OciTarget: False (sidecar-based).

    def exists(self, *, hash: str, client: Any | None) -> bool: ...
        # "Does the known-good complete artifact for this hash exist?"
        # Target-specific — see below.

    async def export(
        self, *, dockerfile: str, tag: str, canonical_json: str,
        client: Any | None, backend: "BuildBackend",
    ) -> Any: ...
        # Solve via BuildKit + write the artifact. Returns the result type.
        # Writes sidecar/label AFTER success (atomic).

    def result_from_cache(self, *, hash: str, client: Any | None) -> Any: ...
        # Reconstruct the result object from an existing artifact (no rebuild).
        # Called after exists() returns True.
```

**`exists(hash, client)`** — target-specific existence check. The content hash is on `ImageSpec` (target-agnostic), but "does the artifact for this hash already exist?" is target-specific:
- `DockerTarget`: `client.images.get(tag)` succeeds → `True`
- `FirecrackerRootfsTarget`: sidecar file `<path>.containerspec.json` exists AND its `hash` field matches → `True`. File-exists alone is not enough (a half-written rootfs from a failed build must trigger rebuild).
- `OciTarget`: sidecar file, same pattern as rootfs

**`export(...)`** — solve via BuildKit and write the artifact. Writes the sidecar/label after success.

**`result_from_cache(...)`** — reconstruct the typed result (`BuiltImage`/`FirecrackerRootfs`/`OciArtifact`) from the existing artifact without rebuilding. For Docker, reads `HF_HOME`/uid from the spec (not the image — the spec is the source of truth). For rootfs/OCI, reads the sidecar.

### Concrete targets

| Target | Result type | Export mechanism | Existence check | Metadata |
|---|---|---|---|---|
| `DockerTarget(*, load=True)` (default; string shortcut) | `BuiltImage` | `buildx --output type=docker` (loads into daemon) | `client.images.get(tag)` | `containerspec.image_spec` Docker label |
| `FirecrackerRootfsTarget(path, *, size_mb=1024)` | `FirecrackerRootfs` | `buildx --output type=local,dest=tmpdir` → `mke2fs -d tmpdir` → ext4 at `path` | sidecar `<path>.containerspec.json` hash match | sidecar `<path>.containerspec.json` |
| `OciTarget(path)` | `OciArtifact` | `buildx --output type=oci,dest=path` | sidecar `<path>.containerspec.json` hash match | sidecar `<path>.containerspec.json` |

**Sidecar metadata file** (`<artifact_path>.containerspec.json`):
```json
{
  "hash": "sha-abc123def4567890...",
  "spec": { ... canonical payload ... }
}
```
For Docker/OCI image targets, the canonical spec is ALSO written as the `containerspec.image_spec` label (image config feature). For rootfs targets, the sidecar is the only metadata home — `containerspec inspect` reads it. The sidecar is written atomically (temp file + rename) after the artifact is complete, so a crash mid-build never leaves a stale sidecar claiming success.

### Result types

```python
@dataclass(frozen=True)
class BuiltImage:
    tag: str
    hf_home: str    # from .env({"HF_HOME": ...}), default "/root/.cache/huggingface"
    uid: int        # from .user(), default 0

@dataclass(frozen=True)
class FirecrackerRootfs:
    path: str       # ext4 image path
    hash: str       # content hash (16-char sha prefix)
    size_mb: int    # actual image size

@dataclass(frozen=True)
class OciArtifact:
    path: str       # OCI tarball path
    hash: str
```

### `BuildBackend` protocol

```python
@runtime_checkable
class BuildBackend(Protocol):
    async def solve_and_export(
        self, *, dockerfile: str, tag: str, output_type: str,
        output_path: str | None, labels: dict[str, str], pull: bool,
    ) -> None: ...
```

| Backend | Mechanism | Targets supported | Notes |
|---|---|---|---|
| `BuildKitBackend(*, builder=None, url=None)` | `docker buildx build --output type=...` via `asyncio.create_subprocess_exec` | all | Default. `url` for remote `buildkitd`. `builder` to pick a named builder. |
| `DockerBackend()` | `docker-py` `images.build()` | Docker only | Fallback when buildx unavailable. Only supports `DockerTarget`. |

`image.build(target, backend=None)` auto-selects: `BuildKitBackend` if `docker buildx` is available, else `DockerBackend` (Docker target only, raises for other targets).

### Layer types (internal)

Frozen dataclasses, discriminated union. The package-manager name IS the layer-type discriminator — adding `dnf_install` later is additive (new type, no cache invalidation of existing specs).

| Layer | Fields | Hash stability |
|---|---|---|
| `AddPython` | `version: str` | — |
| `AptInstall` | `packages: tuple[str, ...]` | sorted within layer |
| `ApkInstall` | `packages: tuple[str, ...]` | sorted within layer |
| `UvPipInstall` | `packages: tuple[str, ...]` | sorted within layer |
| `PipInstall` | `packages: tuple[str, ...]` | sorted within layer |
| `Env` | `vars: Mapping[str, str]` | sorted by key |
| `RunCommands` | `commands: tuple[str, ...]` | order preserved |
| `Workdir` | `path: str` | — |
| `Chown` | `path: str`, `uid: int \| None`, `gid: int \| None` | `None` → resolved at render, `null` in hash |
| `User` | `uid: int`, `gid: int`, `name: str` | — |
| `Entrypoint` | `commands: tuple[str, ...] \| None` | `None` clears, `[]` empties |

v0.1 ships `apt` + `apk` (closed set). Adding `dnf`/`microdnf` later is a new layer type — additive, no existing cache invalidation. The PM-name-as-discriminator property is what makes this safe.

## Content hash

Tag is `{name}:sha-{hash[:16]}` where `hash = sha256(canonical_json)` and `canonical_json` is the JSON serialization of the full readable layer list. **Target-agnostic** — the hash is a function of the spec (content), not the output (packaging). Same spec → same hash, whether the output is a Docker image, a rootfs, or an OCI tarball.

```python
hash_payload = {
    "base": {
        "ref": "nvidia/cuda:13.3.0-devel-ubuntu24.04",
        "digest": "sha256:21f26c94dddeb8f594b3f66a7f7b...",   # resolved (pin_digest=True)
    },
    "pin_digest": true,
    "layers": [
        {"type": "add_python", "version": "3.12"},
        {"type": "apt_install", "packages": ["build-essential", "ffmpeg", "git"]},
        {"type": "uv_pip_install", "packages": ["flashinfer==0.2.0", "vllm"]},
        {"type": "env", "vars": {"HF_HOME": "/home/warden/...", "HF_HUB_...": "1"}},
        {"type": "workdir", "path": "/app"},
        {"type": "chown", "path": "/home/warden/.cache/huggingface", "uid": null, "gid": null},
        {"type": "user", "uid": 1000, "gid": 1000, "name": "warden"},
        {"type": "entrypoint", "commands": []},
    ],
}
```

### Hash properties

- Same spec → same hash, always. Reordering apt/pip packages within a layer doesn't change the hash (sorted). Reordering *layers* does (correct — `chown` before `user` vs after produces a different image).
- `uv_pip_install(["vllm"])` and `pip_install(["vllm"])` produce **different hashes** (different layer type).
- `apt_install(["pkg"])` and `apk_install(["pkg"])` produce **different hashes** (different PM discriminator).
- `.chown(path)` (default) hashes with `uid: null, gid: null` — the hash doesn't change based on which `.user()` precedes it. `.chown(path, uid=0, gid=0)` hashes with `uid: 0, gid: 0`. The payload must distinguish `"uid": 0` from `"uid": null`.
- `pin_digest` choice is part of the hash payload. Switching invalidates cache — correct.
- **The hash is the cache key for content, not for artifacts.** `image.build("warden/vllm")` and `image.build(OciTarget(...))` share a hash but produce two artifacts in two places. Each `BuildTarget.exists(hash)` answers "does THIS artifact for this hash exist?" — the skip-if-built optimization holds per-target, uniformly.

### Base image: tag string vs resolved digest

`from_registry(base, *, pin_digest: bool = True)`.

- **`pin_digest=True` (default):** `content_hash`/`build` calls `client.images.get_registry_data(base).id` to resolve the manifest digest (one HTTP request, **no layer pull** — lazy, at hash/build time). If the base is retagged upstream, the digest changes, the hash changes, a new image builds. Correct.
- **`pin_digest=False`:** the tag string feeds the hash. Faster (no network), cache hits survive a retag. Enables `content_hash(client=None)` and `tag()` without Docker installed.

## Build flow

```python
async def build(self, target: str | BuildTarget, *, client=None, backend=None):
    target = self._resolve_target(target)          # str → DockerTarget()
    if client is None and target.needs_client:
        import docker; client = docker.from_env()
    if backend is None:
        backend = self._auto_backend()             # BuildKitBackend if available, else DockerBackend

    hash_str = self.content_hash(client=client)
    tag = f"{target.name}:{hash_str[:16]}"         # target.name from target or the str

    # 1. Existence check — skip if known-good complete artifact exists.
    if target.exists(hash=hash_str, client=client):
        return target.result_from_cache(hash=hash_str, client=client)

    # 2. Solve + export via BuildKit. Artifact written atomically; sidecar/label last.
    canonical = self._canonical_json(client=client)
    result = await target.export(
        dockerfile=self.to_dockerfile(), tag=tag, canonical_json=canonical,
        client=client, backend=backend,
    )
    return result
```

**Build failure and the cache:** The artifact (image tag / ext4 file / OCI tarball) is only written on full success. `target.exists()` returns `False` for a half-built artifact — a failed build leaves no sidecar and no tag. BuildKit's internal layer cache retains successful layers. The next `build()` resumes from BuildKit's cache. **`exists()` == "known-good complete artifact," never "we started this once."**

### Rootfs creation (`FirecrackerRootfsTarget`)

```python
async def export(self, *, dockerfile, tag, canonical_json, client, backend):
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. BuildKit exports the filesystem tree to tmpdir (no daemon needed).
        await backend.solve_and_export(
            dockerfile=dockerfile, tag=tag,
            output_type="local", output_path=tmpdir, labels={}, pull=True,
        )
        # 2. Create ext4 image populated from tmpdir (no root, no loopback).
        #    Requires e2fsprogs (mke2fs) on the host — fail with clear error if missing.
        self._create_ext4(rootfs_dir=tmpdir, dest=self.path, size_mb=self.size_mb)
    # 3. Write sidecar atomically (temp + rename) — only after ext4 is complete.
    self._write_sidecar(path=f"{self.path}.containerspec.json", hash=tag, spec=canonical_json)
    return FirecrackerRootfs(path=self.path, hash=tag, size_mb=self.size_mb)
```

`mke2fs -d <dir> -t ext4 <dest> <size>` populates an ext4 image from a directory without requiring root or a loopback mount. Requires `e2fsprogs` on the host. If `mke2fs` is not on `$PATH`, `FirecrackerRootfsTarget.export()` raises `MissingToolError("mke2fs not found — install e2fsprogs to build Firecracker rootfs images")` before invoking BuildKit, so the user gets a clear error instead of a confusing BuildKit failure. The `mke2fs` call itself runs via `asyncio.create_subprocess_exec` (non-blocking).

### VM-bootability is spec content, not target config

The `FirecrackerRootfsTarget` handles **output format** (ext4 image creation, size, sidecar). The init system, serial console, and pseudo-fs mounts are **spec layers** — the consumer's adapter composes them. This keeps containerspec pure (no VM knowledge) while fully enabling Firecracker:

```python
# In warden's Firecracker adapter (NOT containerspec):
ImageSpec.from_registry("alpine:3.20")
    .apk_install("openrc", "util-linux")
    .run_commands(
        "ln -s agetty /etc/init.d/agetty.ttyS0",
        "echo ttyS0 > /etc/securetty",
        "rc-update add agetty.ttyS0 default",
        "rc-update add devfs boot",
        "rc-update add procfs boot",
        "rc-update add sysfs boot",
    )
    .build(FirecrackerRootfsTarget("/path/rootfs.ext4", size_mb=2048))
```

## Dockerfile generation

`to_dockerfile()` output for a rootless apt-based spec (Ubuntu/Debian base):

```dockerfile
# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

# add_python("3.12")
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install 3.12 && uv venv --python 3.12 /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# apt_install("build-essential", "ffmpeg", "git")  — sorted
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends build-essential ffmpeg git

# uv_pip_install("flashinfer==0.2.0", "vllm")  — sorted
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    UV_LINK_MODE=copy uv pip install --system flashinfer==0.2.0 vllm

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

For an apk-based spec (Alpine base, Firecracker rootfs use case):

```dockerfile
# syntax=docker/dockerfile:1.7
FROM alpine:3.20

# apk_install("openrc", "util-linux")  — sorted
RUN --mount=type=cache,target=/var/cache/apk,sharing=locked \
    apk add --no-cache openrc util-linux

# run_commands(...)
RUN ln -s agetty /etc/init.d/agetty.ttyS0
RUN echo ttyS0 > /etc/securetty
RUN rc-update add agetty.ttyS0 default

# env({"TERM": "linux"})
ENV TERM=linux

# user(uid=1000, gid=1000, name="warden")
RUN addgroup -g 1000 warden && adduser -u 1000 -G warden -D -h /home/warden warden
USER 1000:1000
```

`--mount=type=cache` directives give BuildKit content-addressable layer caching. `apk_install` renders `apk add --no-cache` with `/var/cache/apk` cache mount (distinct from apt's `/var/cache/apt` + `/var/lib/apt`).

## Dependencies

```toml
[project]
name = "containerspec"
version = "0.1.0"
license = { text = "MIT" }
requires-python = ">=3.12"
dependencies = []  # no required deps — to_dockerfile() and content_hash() work standalone

[project.optional-dependencies]
build = ["docker>=7.0.0"]  # needed for DockerTarget existence check + DockerBackend fallback
```

**Runtime requirements (documented in README):**
- `to_dockerfile()`, `content_hash()`, `tag()` — pure Python, no Docker, no buildx.
- `image.build(DockerTarget)` — requires `docker buildx` CLI (BuildKit solve + `--output type=docker`) AND `docker` Python package (existence check via `client.images.get`).
- `image.build(FirecrackerRootfsTarget)` — requires `docker buildx` CLI AND `e2fsprogs` (`mke2fs`) on host. Does NOT require a Docker daemon (BuildKit can target a remote `buildkitd`).
- `image.build(OciTarget)` — requires `docker buildx` CLI. Does NOT require a Docker daemon.

`docker` (docker-py) is a lazy import inside `build()` when a target needs daemon-side queries. `docker buildx` is invoked via `subprocess` — no Python SDK dependency on BuildKit. This makes the library useful for CI pipelines that generate Dockerfiles for external build systems, and for rootfs/OCI targets that don't need a daemon at all.

## Tests

### Unit tests (no Docker, no buildx — run in default CI)

- **`test_api.py`** — fluent API, immutability, layer types, method chaining, `from_registry` with `pin_digest`. Includes `apk_install` layer.
- **`test_hash.py`** — hash determinism, sensitivity, `pin_digest` lazy resolution, `uid=0` vs `null` in payload, `apt` vs `apk` discriminator. **Golden canonical payload + golden hash** (load-bearing — locks the cache contract).
- **`test_dockerfile.py`** — golden-file snapshots: rootless apt, rootless apk, root specs.
- **`test_chown.py`** — all three chown cases (no user → error, default → resolves to preceding user, explicit override → no user needed).
- **`test_targets.py`** — `BuildTarget` protocol, `exists()` per target (Docker: `images.get`, rootfs/OCI: sidecar hash match), `export()` invocation, `result_from_cache()`.
- **`test_backends.py`** — backend selection, `BuildKitBackend` CLI invocation (mocked `asyncio.create_subprocess_exec`), `DockerBackend` fallback, auto-detect.
- **`test_rootfs.py`** — `mke2fs -d` invocation, sidecar write (atomic temp+rename), `MissingToolError` when `mke2fs` absent.
- **`test_build.py`** — `build()` with mocked backend (async): existence-check skip, export when not found, label/sidecar written, result types.
- **`test_public_api.py`** — `to_dockerfile()`, `content_hash()`, `tag()`, `__repr__`.

### Integration tests (`CONTAINERSPEC_E2E=1` — real BuildKit, gated)

`test_integration.py` — gated behind `CONTAINERSPEC_E2E=1`. Uses `testcontainers` with `moby/buildkit` image to provide a clean, concurrent-safe BuildKit instance. Verifies:
- Docker target: build tiny image from `python:3.12-slim` with `pip_install("httpx")`, second build skips (existence check), `docker run` imports httpx.
- Rootfs target: build ext4 from `alpine:3.20` with `apk_install("openrc")`, verify ext4 mounts and contains `/etc/init.d/`. Second build skips (sidecar hash match).
- OCI target: build OCI tarball, verify `skopeo inspect` or tarball structure.
- Concurrent builds: two different specs build simultaneously against the same BuildKit, no cache collision.

`testcontainers` spins up `moby/buildkit` as a container, exposing `buildkitd` over TCP. `BuildKitBackend(url="tcp://localhost:PORT")` targets it. This gives safe, cache-efficient, concurrent builds in CI without requiring a host-level Docker daemon or buildkitd install.

### Load-bearing tests (write red-then-green first, in UNIT tier — never behind E2E gate)

1. `test_golden_canonical_payload` — locks what's being hashed (by-eye-verifiable JSON). Write first.
2. `test_golden_hash_stable` — locks that the hash over that payload is stable.
3. `test_pin_digest_resolved_digest_feeds_hash` + `test_from_registry_does_not_resolve_until_hash` — locks digest-pinning correctness AND pins resolution as lazy.
4. `test_chown_explicit_uid_zero_in_hash` — locks that explicit `uid=0` hashes as `"uid": 0` not `"uid": null`.
5. `test_target_exists_per_target` — locks that `exists()` is target-specific (Docker checks daemon, rootfs checks sidecar). The cache-skip property holds uniformly, not just for Docker.
6. `test_apt_apk_different_hash` — locks that the PM discriminator works (adding PMs later is additive).

## Tooling

| Tool | Role | Version |
|---|---|---|
| `uv` | Package manager, venv, runner | latest |
| `ruff` | Lint + format (replaces flake8 + black + isort) | latest |
| `pyright` | Type checker (strict over `src/`) — public API is the type surface, don't let a young checker shape it | latest |
| `pytest` + `pytest-asyncio` + `pytest-cov` | Tests, 95% coverage gate | latest |
| `testcontainers` | Integration tests (moby/buildkit container) | latest |
| `mkdocs-material` + `mkdocstrings` | Docs | latest |

`ty` (Astral's new type checker) is promising but pre-1.0 and not yet at feature parity on `Protocol`/generics — and this library's public API *is* its type surface (`BuildTarget`, `BuildBackend` protocols, typed per-target results). Using `pyright` strict avoids "the checker doesn't support this yet" shaping the API.

## Warden integration (separate plan section)

Warden consumes the library via editable path dependency during development, PyPI for release:

```bash
# Development (editable, local path):
cd ~/src/warden
uv add --editable ~/src/containerspec

# After PyPI publish:
uv add containerspec>=0.1.0
```

Two Warden-aware adapters live in `src/warden/backends/base.py`:

1. **`image_spec_from_deploy_spec()`** — composes `containerspec.ImageSpec` with Warden's existing `engine_pip_package()` + `resolve_install_list()`. Output target: `DockerTarget` for `LocalBackend`. `LocalBackend.provision()` auto-builds when `container_image is None and (packages or auto_build)`, mounts uid-namespaced `hf-cache-{uid}` volumes, reads `HF_HOME`/uid from `BuiltImage` (no inspect on hot path). `warden images build/list/inspect` CLI wraps the library.

2. **`firecracker_rootfs_from_deploy_spec()`** — composes `ImageSpec` with VM-bootability layers (openrc, ttyS0, pseudo-fs mounts) for the sandbox Firecracker runtime. Output target: `FirecrackerRootfsTarget`. Same `engine_pip_package()` + `resolve_install_list()` seam, different target + different base image (Alpine for small rootfs).

These are Warden-side changes, not library changes — they compose the library's public API without leaking Warden specifics into it.

## Extraction path

The library is a standalone repo from day one (`github.com/AZX-PBC-OSS/containerspec`), published to PyPI. Warden consumes it as a versioned dependency. The two seams the k8s/Firecracker consumers define (tag namespacing, existence-check generalization to registry manifest HEAD) are cheap to leave as seams now and expensive to retrofit — noted here, not implemented in v0.
