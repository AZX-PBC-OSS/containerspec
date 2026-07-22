# ContainerSpec

Fluent, content-hashed image builder for Docker, Firecracker, and OCI.

ContainerSpec gives you a fluent, immutable `ImageSpec` API that mirrors Modal's
Image builder but targets any Docker daemon, BuildKit instance, or Buildah
(daemonless) for producing Docker images, Firecracker rootfs ext4 images, and
OCI tarballs. Every method returns a new frozen spec, and the spec is
content-hashed so the same configuration never rebuilds from scratch.

## Quick start

Build a Docker image and a Firecracker rootfs from fluent specs. `pin_digest=False`
keeps the examples offline (no registry digest resolution, no `docker` Python
package needed to compute the hash):

```python
import asyncio
from containerspec import ImageSpec, FirecrackerRootfsTarget

# --- Docker image ---
docker_image = (
    ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
    .add_python("3.12")
    .uv_pip_install("vllm", "flashinfer==0.2.0")
    .env({"HF_HOME": "/home/warden/.cache/huggingface"})
    .user(uid=1000, gid=1000, name="warden")
    .chown("/home/warden/.cache/huggingface")
    .entrypoint([])
)

# Pure Python, no Docker required:
print(docker_image.to_dockerfile())
print(docker_image.tag("warden/vllm"))  # warden/vllm:sha-<16 hex chars>

# Build it (requires `pip install containerspec[build]` + docker buildx):
built = asyncio.run(docker_image.build("warden/vllm"))
print(built.tag, built.hf_home, built.uid)

# --- Firecracker rootfs (ext4) ---
rootfs_spec = (
    ImageSpec.from_registry("alpine:3.20", pin_digest=False)
    .apk_install("openrc", "util-linux")
    .run_commands("ln -s agetty /etc/init.d/agetty.ttyS0")
    .env({"TERM": "linux"})
    .user(uid=1000, gid=1000, name="warden")
    .entrypoint([])
)

rootfs = asyncio.run(
    rootfs_spec.build(FirecrackerRootfsTarget(path="./rootfs.ext4", size_mb=256))
)
print(rootfs.path, rootfs.hash, rootfs.size_mb)
```

## Why

Building images for different package combinations (vLLM + flashinfer + torch
pins, SGLang + custom kernels, Alpine + openrc rootfs images, Rust CLI tools,
Node frontends, etc.) is repetitive. ContainerSpec gives you a fluent API that
mirrors Modal's Image builder but targets any Docker daemon, BuildKit instance,
or Buildah — and content-hashes the spec so the same configuration never
rebuilds from scratch.

Two specs that build the same layers in the same order produce the same content
hash, regardless of output target. The hash is the cache key: `build()` skips
the build entirely when the artifact already exists.

## Install

```bash
pip install containerspec          # spec generation, hashing, dockerfile rendering
pip install containerspec[build]   # + Docker image building (docker-py)
```

Requires Python 3.12+. The `build` extra installs `docker>=7.0.0` (docker-py),
needed for `DockerTarget` existence checks and `pin_digest=True` digest
resolution. See [Installation](https://AZX-PBC-OSS.github.io/containerspec/getting-started/installation/)
for the full runtime requirements matrix (buildx, buildah, e2fsprogs, oci2rootfs).

## API reference

### `ImageSpec` methods

Every method returns a new frozen `ImageSpec` (the original is never mutated).
Install/package layers sort their inputs within the layer so reordering
arguments does not change the hash; `run_commands` preserves order. Methods that
take packages raise `ValueError` if called with no arguments.

#### Construction

| Method | Returns | Notes |
|--------|---------|-------|
| `from_registry(base, *, pin_digest=True)` | `ImageSpec` | Classmethod. Start of every chain. `pin_digest=True` pins `FROM` to the registry digest (requires a docker client at build/hash time). Raises `ValueError` if `base` is empty/whitespace. |
| `with_stage(name)` | `StageSpec` | Create a named build stage from this spec for use in multi-stage builds. See [Multi-stage builds](#multi-stage-builds). |

#### Package managers / system installs

| Method | Returns | Notes |
|--------|---------|-------|
| `apt_install(*packages)` | `ImageSpec` | Debian/Ubuntu. Packages sorted. Uses an apt cache mount (`--mount=type=cache,target=/var/cache/apt`). |
| `apk_install(*packages)` | `ImageSpec` | Alpine. Packages sorted. `apk add --no-cache`. |
| `dnf_install(*packages)` | `ImageSpec` | RPM distros (RHEL, Fedora, UBI). Packages sorted. Uses a dnf cache mount. |
| `brew_install(*packages)` | `ImageSpec` | Homebrew/Linuxbrew. Installs brew if missing. Packages sorted. |
| `add_python(version)` | `ImageSpec` | Installs Python via `uv` into `/opt/venv` and prepends it to `PATH`. Copies `uv`/`uvx` from `ghcr.io/astral-sh/uv:latest`. |
| `uv_pip_install(*packages)` | `ImageSpec` | Fast pip via `uv pip install --system` with a uv cache mount. Packages sorted. |
| `pip_install(*packages)` | `ImageSpec` | Standard `pip install`. Packages sorted. |

#### Language toolchains

| Method | Returns | Notes |
|--------|---------|-------|
| `nvm_install(version)` | `ImageSpec` | Installs Node.js via nvm. Symlinks node/npm/npx to `/usr/local/bin`. |
| `npm_install(*packages)` | `ImageSpec` | `npm install -g`. Packages sorted. Requires node on PATH (e.g. after `nvm_install`). |
| `pnpm_install(*packages)` | `ImageSpec` | Installs pnpm globally, then `pnpm add -g`. Packages sorted. pnpm store is cache-mounted. |
| `rust_install()` | `ImageSpec` | Installs Rust via rustup. Sets `PATH` for `cargo`/`rustc`. |
| `cargo_install(*packages)` | `ImageSpec` | `cargo install`. Packages sorted. Requires rust on PATH (e.g. after `rust_install`). Cargo registry + git are cache-mounted. |
| `uvx_install(*packages)` | `ImageSpec` | Run tools via `uvx --system`. Packages sorted. uv cache is cache-mounted. |

#### Filesystem / metadata

| Method | Returns | Notes |
|--------|---------|-------|
| `env(vars)` | `ImageSpec` | `ENV`. Keys sorted for hash stability. Pass a mapping. |
| `run_commands(*commands)` | `ImageSpec` | One `RUN` per command. Order preserved. |
| `workdir(path)` | `ImageSpec` | `WORKDIR`. |
| `chown(path, *, uid=None, gid=None)` | `ImageSpec` | `chown -R`. Both uid/gid must be set, or both omitted. If omitted, resolved from the nearest preceding `.user()` layer (raises `ValueError` if there is none). Renders a `USER root` sandwich when a non-root user is active. |
| `user(*, uid, gid, name)` | `ImageSpec` | Creates group + user and sets `USER uid:gid`. |
| `entrypoint(commands)` | `ImageSpec` | `ENTRYPOINT`. `None` omits it; `[]` renders `ENTRYPOINT []`. |
| `expose(*ports)` | `ImageSpec` | `EXPOSE`. Ports sorted. |
| `cmd(commands)` | `ImageSpec` | `CMD`. `None` omits it; `[]` renders `CMD []`. |
| `volume(*paths)` | `ImageSpec` | `VOLUME`. Paths sorted. |
| `copy(src, dest, *, content_hash=None)` | `ImageSpec` | `COPY src dest` with content hashing. See [copy() with content hashing](#copy-with-content-hashing). |
| `copy_from_stage(stage, src, dest)` | `ImageSpec` | `COPY --from=<stage>`. See [Multi-stage builds](#multi-stage-builds). |

#### Introspection / build

| Method | Returns | Notes |
|--------|---------|-------|
| `to_dockerfile()` | `str` | Renders a Dockerfile. Pure — no Docker needed. Uses the base tag as-is (digest pinning happens at build time). |
| `content_hash(*, client=None)` | `str` | Full sha256 hex (64 chars) of the canonical payload. Target-agnostic. Requires `client` when `pin_digest=True`. |
| `tag(name, *, client=None)` | `str` | `{name}:sha-{hash[:16]}` without building. |
| `build(target, *, client=None, backend=None)` | awaitable | **Async.** Builds a Docker image, rootfs, or OCI tarball. Skips the build when the artifact already exists. See [Build targets](#build-targets) and [Error handling](#error-handling-and-debugging). |

### `StageSpec`

A named build stage for multi-stage Dockerfiles, created by `with_stage()`:

```python
@dataclass(frozen=True)
class StageSpec:
    name: str
    spec: ImageSpec
```

Pass a `StageSpec` to `copy_from_stage()` to compose a build stage into a
runtime image. The stage's content hash is included in the canonical payload,
so changes to the stage's spec bust the cache.

## Build targets

`build()` accepts a target that determines the output format and result type.
Passing a bare string is a shortcut for `DockerTarget(name=...)`.

| Target | Constructor | Result type | Fields |
|--------|-------------|-------------|--------|
| `DockerTarget` | `DockerTarget(name)` (or `build("name")`) | `BuiltImage` | `tag`, `hf_home`, `uid` |
| `FirecrackerRootfsTarget` | `FirecrackerRootfsTarget(path, size_mb=1024, converter="mke2fs", converter_image="oci2rootfs:latest")` | `FirecrackerRootfs` | `path`, `hash`, `size_mb` |
| `OciTarget` | `OciTarget(path)` | `OciArtifact` | `path`, `hash` |

`BuiltImage.hf_home` and `BuiltImage.uid` are enriched from the spec: `hf_home`
comes from the `HF_HOME` env var (default `/root/.cache/huggingface`), `uid`
from the last `.user()` layer (default `0`).

`FirecrackerRootfsTarget` and `OciTarget` write a sidecar
`<path>.containerspec.json` recording the hash and canonical spec, so
subsequent builds skip when the hash matches.

`converter` selects how the Firecracker rootfs ext4 is produced:

- `"mke2fs"` (default) — builds a local filesystem export, then packs it into
  ext4 with `mke2fs -d`. Requires `e2fsprogs` on the host. No Docker daemon.
- `"oci2rootfs"` — builds an OCI tarball, then converts it to ext4 inside an
  `oci2rootfs` container via `docker run`. Requires a Docker daemon and the
  `converter_image` (default `oci2rootfs:latest`). No host `e2fsprogs`. Handles
  full OCI whiteout semantics.

## Build backends

The backend executes the build. If you do not pass `backend=` to `build()`,
`auto_detect_backend(target=...)` picks one:

| Backend | How it builds | Default for | Notes |
|---------|---------------|-------------|-------|
| `BuildKitBackend` | `docker buildx build` CLI | `DockerTarget` (when `docker` is on PATH) | Supports all output types (`docker`, `oci`, `local`). Constructor: `BuildKitBackend(url=None, builder=None)`. |
| `BuildahBackend` | `buildah bud` + `buildah push` | `FirecrackerRootfsTarget`, `OciTarget` (when `buildah` is on PATH) | Daemonless. Does **not** support `output_type="docker"`. Linux-only. |
| `DockerBackend` | docker-py `client.images.build` | `DockerTarget` fallback (no `docker` CLI) | Docker target only; raises `BuildError` for non-Docker output types. Constructor: `DockerBackend(client=None)`. |

For non-Docker targets, `auto_detect_backend` prefers `BuildahBackend`, then
`BuildKitBackend`, then `DockerBackend` (which will error for non-Docker
output types — install `buildah` or `docker buildx`).

## Multi-stage builds

Use `with_stage()` to create a named build stage, then `copy_from_stage()` to
pull artifacts from that stage into a runtime image. The stage is rendered as a
`FROM ... AS <name>` block before the runtime `FROM`; duplicate stage names are
ignored so the Dockerfile emits a single `FROM ... AS <name>` per stage.

A common pattern is a heavy builder stage (compilers, dev dependencies) copied
into a slim runtime image:

```python
import asyncio
from containerspec import ImageSpec

# Builder stage: node toolchain that builds the frontend.
builder_stage = (
    ImageSpec.from_registry("node:22", pin_digest=False)
    .workdir("/app")
    .run_commands("npm install", "npm run build")
    .with_stage("builder")
)

# Runtime stage: nginx serving the built artifacts.
runtime = (
    ImageSpec.from_registry("nginx:alpine", pin_digest=False)
    .copy_from_stage(builder_stage, "/app/dist", "/usr/share/nginx/html")
    .expose(80)
    .entrypoint(["nginx", "-g", "daemon off;"])
)

print(runtime.to_dockerfile())
# Build it:
# built = asyncio.run(runtime.build("myapp"))
```

Rendered Dockerfile (note the `FROM ... AS builder` block precedes the runtime):

```dockerfile
# syntax=docker/dockerfile:1.7
FROM node:22 AS builder

# workdir("/app")
WORKDIR /app

# run_commands("npm install", "npm run build")
RUN npm install
RUN npm run build


FROM nginx:alpine

# copy_from_stage("builder", "/app/dist", "/usr/share/nginx/html")
COPY --from=builder /app/dist /usr/share/nginx/html

# expose(80)
EXPOSE 80

# entrypoint(["nginx", "-g", "daemon off;"])
ENTRYPOINT ["nginx", "-g", "daemon off;"]
```

Multiple stages are supported — call `copy_from_stage()` once per stage. Changes
to a stage's spec change its hash, which busts the runtime cache:

```python
from containerspec import ImageSpec

builder_a = ImageSpec.from_registry("node:22", pin_digest=False).workdir("/app")
builder_b = (
    ImageSpec.from_registry("node:22", pin_digest=False)
    .workdir("/app")
    .run_commands("npm run build")  # extra step
)

stage_a = builder_a.with_stage("builder")
stage_b = builder_b.with_stage("builder")

runtime_a = ImageSpec.from_registry("nginx:alpine", pin_digest=False).copy_from_stage(
    stage_a, "/app/dist", "/usr/share/nginx/html"
)
runtime_b = ImageSpec.from_registry("nginx:alpine", pin_digest=False).copy_from_stage(
    stage_b, "/app/dist", "/usr/share/nginx/html"
)

assert runtime_a.content_hash(client=None) != runtime_b.content_hash(client=None)
```

## copy() with content hashing

`copy()` includes the source content's hash in the canonical payload, so
changing the copied file busts the cache automatically. Two modes:

**1. Local file exists** — `copy()` hashes the file (or directory, recursively)
for you:

```python
from containerspec import ImageSpec

# ./app.py exists on disk; its content is hashed into the spec.
spec = ImageSpec.from_registry("python:3.12-slim", pin_digest=False).copy(
    "./app.py", "/app/app.py"
)
print(spec.layers[0].content_hash)  # sha256 hex of ./app.py contents
```

Editing `./app.py` and rebuilding produces a different hash, so `build()`
rebuilds instead of hitting the cache.

**2. User-provided hash** — for CI pipelines where the file isn't local but its
hash is known (e.g. an artifact fetched from a remote store):

```python
from containerspec import ImageSpec

spec = ImageSpec.from_registry("python:3.12-slim", pin_digest=False).copy(
    "remote://artifact.tar",
    "/app/artifact.tar",
    content_hash="sha256:abc123def456",
)
```

If the source path does not exist **and** no `content_hash` is provided,
`copy()` raises `FileNotFoundError` with a message pointing at the
`content_hash=` escape hatch:

```python
from containerspec import ImageSpec

try:
    ImageSpec.from_registry("base", pin_digest=False).copy("/nonexistent", "/app/x")
except FileNotFoundError as e:
    print(e)
    # copy() source '/nonexistent' does not exist. Either provide a local path
    # that exists, or pass content_hash= explicitly: copy('/nonexistent',
    # '/app/x', content_hash='sha256:...')
```

## Context-aware rendering

Dockerfile is a stateful, sequential format — `USER`, `WORKDIR`, and `ENV`
persist for all subsequent layers. ContainerSpec tracks this accumulated state
(`RenderContext`) so each layer renders with correct paths, cache mounts, and
user switches. You do not have to think about it; the rendered Dockerfile is
correct for the state at each layer.

**Cache mounts follow the active USER.** Tool cache targets are computed from
the current `HOME`, which is `/home/<name>` when a non-root user is active and
`/root` otherwise:

```python
from containerspec import ImageSpec

spec = (
    ImageSpec.from_registry("base", pin_digest=False)
    .user(uid=1000, gid=1000, name="warden")
    .uv_pip_install("httpx")
    .rust_install()
    .cargo_install("ripgrep")
)
df = spec.to_dockerfile()
# uv cache -> /home/warden/.cache/uv  (not /root/.cache/uv)
# rust PATH -> /home/warden/.cargo/bin
# cargo registry -> /home/warden/.cargo/registry
```

**chown gets a USER root sandwich.** Dockerfile's `USER` directive persists, so
`RUN chown` fails with `EPERM` under a non-root user. ContainerSpec temporarily
switches to `USER root`, runs the `chown`, then switches back:

```python
from containerspec import ImageSpec

spec = (
    ImageSpec.from_registry("base", pin_digest=False)
    .user(uid=1000, gid=1000, name="warden")
    .chown("/data")
)
df = spec.to_dockerfile()
```

```dockerfile
# ...
# user(uid=1000, gid=1000, name="warden")
RUN groupadd -g 1000 warden && useradd -u 1000 -g 1000 -m -d /home/warden warden
USER 1000:1000

# chown("/data") — resolved to uid=1000, gid=1000 from preceding .user()
USER root
RUN mkdir -p /data && chown -R 1000:1000 /data
USER 1000:1000
```

The same `chown` under root (or before any `.user()`) renders a plain
`RUN ... chown` with no sandwich.

## Runtime requirements

| Operation | Requires | Docker daemon? |
|-----------|----------|----------------|
| `to_dockerfile()`, `content_hash()`, `tag()` | Pure Python (no Docker, no buildx) | No |
| `build(str)` / `build(DockerTarget)` | `containerspec[build]` (docker-py) + `docker buildx` CLI (docker-py build is the fallback) | Yes (image is loaded into the daemon) |
| `build(FirecrackerRootfsTarget)` with `converter="mke2fs"` | `e2fsprogs` (`mke2fs`) + a build backend (`buildah` or `docker buildx`) | No |
| `build(FirecrackerRootfsTarget)` with `converter="oci2rootfs"` | Docker daemon + `oci2rootfs` container image | Yes |
| `build(OciTarget)` | `docker buildx` CLI or `buildah` | No |

Notes:

- `pin_digest=True` (the default) resolves the base image digest via docker-py
  (`client.images.get_registry_data`), so it needs the `docker` Python
  package. Use `pin_digest=False` to hash/render without docker-py.
- `build()` is `async` — `await` it or wrap with `asyncio.run()`.
- `MissingToolError` is raised when `mke2fs` or `buildah` is required but not
  on `PATH`. `BuildError` is raised when a build subprocess fails (see
  [Error handling](#error-handling-and-debugging)).

## Firecracker rootfs example

A minimal Alpine rootfs with openrc and a serial console on ttyS0:

```python
import asyncio
from containerspec import ImageSpec, FirecrackerRootfsTarget

spec = (
    ImageSpec.from_registry("alpine:3.20", pin_digest=False)
    .apk_install("openrc", "util-linux")
    .run_commands("ln -s agetty /etc/init.d/agetty.ttyS0")
    .env({"TERM": "linux"})
    .user(uid=1000, gid=1000, name="warden")
    .entrypoint([])
)

# mke2fs converter (default): needs e2fsprogs on the host, no Docker daemon.
rootfs = asyncio.run(
    spec.build(FirecrackerRootfsTarget(path="./rootfs.ext4", size_mb=256))
)

# oci2rootfs converter: needs a Docker daemon + an oci2rootfs image, no host e2fsprogs.
rootfs = asyncio.run(
    spec.build(
        FirecrackerRootfsTarget(
            path="./rootfs.ext4",
            size_mb=256,
            converter="oci2rootfs",
            converter_image="oci2rootfs:latest",
        )
    )
)
```

The rendered Dockerfile:

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
RUN groupadd -g 1000 warden && useradd -u 1000 -g 1000 -m -d /home/warden warden
USER 1000:1000

# entrypoint([])
ENTRYPOINT []
```

## Custom registries

`from_registry` accepts any image reference, including private and authenticated
registries:

```python
ImageSpec.from_registry("ghcr.io/owner/image:tag")
ImageSpec.from_registry("myacr.azurecr.io/image:tag")
ImageSpec.from_registry("registry.example.com:5000/team/app:latest")
```

With `pin_digest=True`, the digest is resolved through docker-py, which reads
your existing Docker config (`~/.docker/config.json`) for registry credentials.
Any registry you can `docker pull` works with `pin_digest=True`.

## Error handling and debugging

`build()` wraps build failures in a `BuildError`. The error message records the
failed target tag and the path where the failed Dockerfile was saved for
debugging:

```python
import asyncio
from containerspec import ImageSpec, BuildError

spec = (
    ImageSpec.from_registry("alpine:3.20", pin_digest=False)
    .apk_install("this-package-definitely-does-not-exist-xyz")
)

try:
    asyncio.run(spec.build("myapp"))
except BuildError as e:
    print(e)
    # Build failed for myapp:sha-<hash>. Failed Dockerfile saved to
    # /tmp/containerspec-failed-<hash>.Dockerfile
```

`BuildError` exposes `.cmd` and `.stderr` for programmatic diagnostics. When a
backend subprocess fails, the backend raises a `BuildError` with `cmd` (the
full command list) and `stderr` (the captured build log) populated. `build()`
catches that and re-raises a new `BuildError` (with the saved-Dockerfile path in
its message) chaining the original as `__cause__` — so reach the populated
diagnostics via the cause:

```python
try:
    asyncio.run(spec.build("myapp"))
except BuildError as e:
    cause = e.__cause__          # the backend BuildError
    if isinstance(cause, BuildError):
        print("cmd:", cause.cmd)       # e.g. ["docker", "buildx", "build", ...]
        print("stderr:", cause.stderr)  # the build log
```

The saved Dockerfile lets you reproduce the failure directly:

```bash
docker buildx build -f /tmp/containerspec-failed-<hash>.Dockerfile .
```

`BuildError` is also raised by backends for unsupported output types (e.g.
`BuildahBackend` with `output_type="docker"`, or `DockerBackend` with a
non-Docker target). `MissingToolError` is raised when `mke2fs` or `buildah` is
required but not on `PATH`.

## License

MIT
