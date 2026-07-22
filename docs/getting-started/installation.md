# Installation

```bash
pip install containerspec          # spec generation, hashing, dockerfile rendering
pip install containerspec[build]   # + Docker image building (docker-py)
```

Requires Python 3.12+. The `build` extra installs `docker>=7.0.0` (docker-py),
needed for `DockerTarget` existence checks and `pin_digest=True` digest
resolution.

## Runtime requirements

ContainerSpec splits cleanly between pure-Python spec operations and build
operations. You can generate Dockerfiles and content hashes with nothing
installed but ContainerSpec itself; building requires external tools depending
on the target and backend.

| Operation | Requires | Docker daemon? |
|-----------|----------|----------------|
| `to_dockerfile()`, `content_hash()`, `tag()` | Pure Python (no Docker, no buildx) | No |
| `build(str)` / `build(DockerTarget)` | `containerspec[build]` (docker-py) + `docker buildx` CLI (docker-py build is the fallback) | Yes (image is loaded into the daemon) |
| `build(FirecrackerRootfsTarget)` with `converter="mke2fs"` | `e2fsprogs` (`mke2fs`) + a build backend (`buildah` or `docker buildx`) | No |
| `build(FirecrackerRootfsTarget)` with `converter="oci2rootfs"` | Docker daemon + `oci2rootfs` container image (default `oci2rootfs:latest`) | Yes |
| `build(OciTarget)` | `docker buildx` CLI or `buildah` | No |

### Build backends

The backend executes the build. If you do not pass `backend=` to `build()`,
`auto_detect_backend(target=...)` picks one based on what is on `PATH`:

| Backend | How it builds | Default for | Requires |
|---------|---------------|-------------|----------|
| `BuildKitBackend` | `docker buildx build` CLI | `DockerTarget` (when `docker` is on PATH) | `docker` CLI with buildx |
| `BuildahBackend` | `buildah bud` + `buildah push` | `FirecrackerRootfsTarget`, `OciTarget` (when `buildah` is on PATH) | `buildah` (Linux, daemonless) |
| `DockerBackend` | docker-py `client.images.build` | `DockerTarget` fallback (no `docker` CLI) | `containerspec[build]` (docker-py) |

For non-Docker targets, `auto_detect_backend` prefers `BuildahBackend`, then
`BuildKitBackend`, then `DockerBackend` (which will error for non-Docker output
types — install `buildah` or `docker buildx`).

### Firecracker rootfs converters

`FirecrackerRootfsTarget` supports two `converter` modes for producing the ext4
image:

- `"mke2fs"` (default) — builds a local filesystem export, then packs it into
  ext4 with `mke2fs -d`. Requires `e2fsprogs` on the host. No Docker daemon.
- `"oci2rootfs"` — builds an OCI tarball, then converts it to ext4 inside an
  `oci2rootfs` container via `docker run`. Requires a Docker daemon and the
  `converter_image` (default `oci2rootfs:latest`). No host `e2fsprogs`. Handles
  full OCI whiteout semantics. Build an `oci2rootfs` image from
  [oci2rootfs](https://github.com/arcboxlabs/oci2rootfs) or provide your own via
  `converter_image=`.

## Notes

- `pin_digest=True` (the default) resolves the base image digest via docker-py
  (`client.images.get_registry_data`), so it needs the `docker` Python package
  at hash/build time. Use `pin_digest=False` to hash and render Dockerfiles
  without docker-py.
- `build()` is `async` — `await` it or wrap with `asyncio.run()`.
- `MissingToolError` is raised when `mke2fs` or `buildah` is required but not
  on `PATH`. `BuildError` is raised when a build subprocess fails (it includes
  `cmd` and `stderr` on the backend-level error, accessible via the chained
  cause from `build()`).

## Installing host tools

```bash
# e2fsprogs (for Firecracker rootfs mke2fs converter)
sudo apt-get install e2fsprogs          # Debian/Ubuntu
sudo dnf install e2fsprogs              # Fedora/RHEL

# buildah (daemonless builds for rootfs/OCI — BuildahBackend)
sudo apt-get install buildah            # Debian/Ubuntu
sudo dnf install buildah                # Fedora/RHEL

# docker buildx (default backend for Docker targets — BuildKitBackend)
# Install Docker Engine, which ships buildx: https://docs.docker.com/engine/install/

# oci2rootfs container image (for Firecracker rootfs oci2rootfs converter)
# Build from source: https://github.com/arcboxlabs/oci2rootfs
docker build -t oci2rootfs:latest https://github.com/arcboxlabs/oci2rootfs.git
```
