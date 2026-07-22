# Quick Start

This page builds a Docker image, a Firecracker rootfs, and an OCI tarball from a
single `ImageSpec`, then covers multi-stage builds, content-hashed `copy()`, and
error handling.

## Install

```bash
pip install containerspec[build]   # spec + Docker image building (docker-py)
```

See [Installation](installation.md) for the full runtime requirements matrix
(buildx, buildah, e2fsprogs, oci2rootfs).

## One spec, three targets

`ImageSpec` is target-agnostic: the same spec can be built as a Docker image, a
Firecracker rootfs, or an OCI tarball. The content hash is identical across all
three, so each target caches independently.

```python
import asyncio
from containerspec import (
    ImageSpec,
    DockerTarget,
    FirecrackerRootfsTarget,
    OciTarget,
)

spec = (
    ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
    .pip_install("httpx")
    .env({"PYTHONUNBUFFERED": "1"})
    .entrypoint([])
)

# 1. Docker image — loaded into the Docker daemon.
image = asyncio.run(spec.build("myapp"))
print(image.tag)  # myapp:sha-<16 hex chars>

# 2. OCI tarball — no Docker daemon needed (docker buildx or buildah).
oci = asyncio.run(spec.build(OciTarget(path="./myapp.tar")))
print(oci.path, oci.hash)

# 3. Firecracker rootfs — ext4 image, no Docker daemon (mke2fs converter).
rootfs = asyncio.run(
    spec.build(FirecrackerRootfsTarget(path="./rootfs.ext4", size_mb=256))
)
print(rootfs.path, rootfs.hash, rootfs.size_mb)
```

`FirecrackerRootfsTarget` also supports `converter="oci2rootfs"` (needs a Docker
daemon + an `oci2rootfs` container image, no host `e2fsprogs`) — see
[Installation](installation.md).

## Multi-stage builds

Use `with_stage()` to name a build stage and `copy_from_stage()` to pull
artifacts from it into a runtime image. A heavy builder stage (compilers, dev
dependencies) copied into a slim runtime image is the common pattern:

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

# Render the multi-stage Dockerfile without Docker:
print(runtime.to_dockerfile())

# Build it:
# built = asyncio.run(runtime.build("myapp"))
```

The stage's content hash feeds the runtime's hash, so changing the builder spec
(e.g. adding a `run_commands` step) busts the runtime cache automatically.
Multiple stages are supported — call `copy_from_stage()` once per stage.

## copy() with content hashing

`copy()` hashes the source content into the canonical payload, so editing a
copied file and rebuilding produces a new hash (no stale cache). For local files
the hash is computed for you:

```python
from containerspec import ImageSpec

# ./app.py exists on disk; its content is hashed into the spec.
spec = ImageSpec.from_registry("python:3.12-slim", pin_digest=False).copy(
    "./app.py", "/app/app.py"
)
```

For CI pipelines where the file isn't local but its hash is known, pass
`content_hash=` explicitly:

```python
spec = ImageSpec.from_registry("python:3.12-slim", pin_digest=False).copy(
    "remote://artifact.tar",
    "/app/artifact.tar",
    content_hash="sha256:abc123def456",
)
```

If the source path does not exist and no `content_hash` is provided, `copy()`
raises `FileNotFoundError` pointing at the `content_hash=` escape hatch.

## Render a Dockerfile without Docker

`to_dockerfile()`, `content_hash()`, and `tag()` are pure Python — no Docker
daemon, no `docker buildx`, no `docker` Python package required (with
`pin_digest=False`):

```python
from containerspec import ImageSpec

spec = (
    ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
    .add_python("3.12")
    .uv_pip_install("vllm", "flashinfer==0.2.0")
    .env({"HF_HOME": "/home/warden/.cache/huggingface"})
    .user(uid=1000, gid=1000, name="warden")
    .chown("/home/warden/.cache/huggingface")
    .entrypoint([])
)

print(spec.to_dockerfile())
print(spec.tag("warden/vllm"))  # warden/vllm:sha-<16 hex chars>
```

## Caching

`build()` computes the content hash and skips the build when the artifact
already exists. For `DockerTarget` it checks the Docker daemon for an image
tagged `{name}:sha-{hash[:16]}`; for `FirecrackerRootfsTarget` and `OciTarget`
it checks a sidecar file (`<path>.containerspec.json`) recording the hash.

```python
# First call builds; second call with the same spec is a cache hit and skips.
built1 = asyncio.run(spec.build("myapp"))
built2 = asyncio.run(spec.build("myapp"))
assert built2.tag == built1.tag
```

## Error handling

`build()` wraps build failures in a `BuildError`. The message records the failed
target tag and the path where the failed Dockerfile was saved for debugging. The
populated `cmd`/`stderr` diagnostics are on the chained cause (the backend
`BuildError`):

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
    print(e)              # Build failed for myapp:sha-<hash>. Failed Dockerfile
                          # saved to /tmp/containerspec-failed-<hash>.Dockerfile
    cause = e.__cause__   # the backend BuildError
    if isinstance(cause, BuildError):
        print(cause.cmd)     # full command list, e.g. ["docker", "buildx", ...]
        print(cause.stderr)  # captured build log
```

Reproduce the failure directly with the saved Dockerfile:

```bash
docker buildx build -f /tmp/containerspec-failed-<hash>.Dockerfile .
```

## `build()` is async

`ImageSpec.build()` is a coroutine — `await` it from an async context or wrap
with `asyncio.run()`. Concurrent builds with different specs do not collide
because each gets its own hash-derived tag.

```python
import asyncio
from containerspec import ImageSpec

spec_a = ImageSpec.from_registry("python:3.12-slim", pin_digest=False).pip_install("httpx")
spec_b = ImageSpec.from_registry("python:3.12-slim", pin_digest=False).pip_install("requests")

a, b = asyncio.run(asyncio.gather(spec_a.build("myapp"), spec_b.build("myapp")))
assert a.tag != b.tag
```
