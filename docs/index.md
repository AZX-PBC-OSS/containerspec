# containerspec

Fluent, content-hashed image builder for Docker, Firecracker, and OCI.

ContainerSpec gives you a fluent, immutable `ImageSpec` API that mirrors Modal's
Image builder but targets any Docker daemon, BuildKit instance, or Buildah
(daemonless) for producing Docker images, Firecracker rootfs ext4 images, and
OCI tarballs. Every method returns a new frozen spec, and the spec is
content-hashed so the same configuration never rebuilds from scratch.

## Features

- **Fluent, immutable `ImageSpec`** — every method returns a new frozen spec.
- **Three output targets** — Docker image, Firecracker rootfs ext4, OCI tarball.
- **Three build backends** — `docker buildx` (BuildKit), `buildah` (daemonless),
  docker-py (fallback).
- **Multi-stage builds** — `with_stage()` + `copy_from_stage()` compose a build
  stage into a runtime image; stage changes bust the runtime cache.
- **Content-hashed `copy()`** — local files are hashed automatically; CI
  pipelines can pass `content_hash=` for remote artifacts.
- **All package managers** — apt, apk, dnf, brew, pip, uv pip, plus language
  toolchains: nvm/npm/pnpm (Node), rustup/cargo (Rust), uvx (Python tools).
- **Context-aware rendering** — cache mounts follow the active `USER`; `chown`
  gets a `USER root` sandwich under non-root users.
- **Content-hash caching** — same spec, same hash, no rebuild. The hash is the
  cache key across all targets.
- **Error diagnostics** — `BuildError` carries `cmd`/`stderr`; failed
  Dockerfiles are saved to `/tmp` for reproduction.

## Next steps

- [Installation](getting-started/installation.md) — runtime requirements matrix.
- [Quick Start](getting-started/quick-start.md) — Docker, OCI, Firecracker,
  multi-stage, and error handling examples.
- [API Reference](api-reference/containerspec.md) — auto-generated from
  docstrings.
