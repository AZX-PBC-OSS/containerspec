# Changelog

## [0.1.7](https://github.com/AZX-PBC-OSS/containerspec/compare/v0.1.6...v0.1.7) (2026-07-22)


### Continuous Integration

* local self-contained publish workflow with attestations off ([#26](https://github.com/AZX-PBC-OSS/containerspec/issues/26)) ([861f9d6](https://github.com/AZX-PBC-OSS/containerspec/commit/861f9d6e44d189eec9bed26914a5b7f8826c93ed))

## [0.1.6](https://github.com/AZX-PBC-OSS/containerspec/compare/v0.1.5...v0.1.6) (2026-07-22)


### Bug Fixes

* reject shell-unsafe input in render sinks (user/chown/version/copy/env/FROM) ([#12](https://github.com/AZX-PBC-OSS/containerspec/issues/12)) ([9eba8de](https://github.com/AZX-PBC-OSS/containerspec/commit/9eba8de40baa9fd6fc60afe67429b9e0e8fd76e6))
* use valid buildah push argv for OCI export ([#14](https://github.com/AZX-PBC-OSS/containerspec/issues/14)) ([79be14e](https://github.com/AZX-PBC-OSS/containerspec/commit/79be14eed447e3d9cf1d22119d890d6ecf7fb119))

## [0.1.5](https://github.com/AZX-PBC-OSS/containerspec/compare/v0.1.4...v0.1.5) (2026-07-22)


### Bug Fixes

* rootless cache mounts + pnpm/yarn bootstrap ([f182168](https://github.com/AZX-PBC-OSS/containerspec/commit/f1821689c9a3f1549325d8d2dd2bc64be376dd71))
* UV_PYTHON_INSTALL_DIR=/opt/uv-python for rootless venv access ([f634537](https://github.com/AZX-PBC-OSS/containerspec/commit/f6345378dc3ab0ebe9f387eeac5dd9e737d984a3))

## [0.1.4](https://github.com/AZX-PBC-OSS/containerspec/compare/v0.1.3...v0.1.4) (2026-07-22)


### Bug Fixes

* idempotent user/group creation (groupadd/useradd || true) ([626fc30](https://github.com/AZX-PBC-OSS/containerspec/commit/626fc301e4c52c071a36433c4b18e8c0bcabae46))

## [0.1.3](https://github.com/AZX-PBC-OSS/containerspec/compare/v0.1.2...v0.1.3) (2026-07-22)


### Bug Fixes

* eliminate shell from package install renderers (exec-form RUN) ([b730984](https://github.com/AZX-PBC-OSS/containerspec/commit/b730984025c3dc75ae1284d1c3eb6e4e979abbda))

## [0.1.2](https://github.com/AZX-PBC-OSS/containerspec/compare/v0.1.1...v0.1.2) (2026-07-22)


### Bug Fixes

* pass the staged Dockerfile to backends explicitly ([#4](https://github.com/AZX-PBC-OSS/containerspec/issues/4)) ([9b06c6a](https://github.com/AZX-PBC-OSS/containerspec/commit/9b06c6a51c2a3ebe65e0fb76bd3e44aa22606681))

## [0.1.1](https://github.com/AZX-PBC-OSS/containerspec/compare/v0.1.0...v0.1.1) (2026-07-22)


### Bug Fixes

* build context staging for copy() — no more broken multi-stage builds ([a368a40](https://github.com/AZX-PBC-OSS/containerspec/commit/a368a4044961e77be5c53ef1608f7113030ad72f))
* build context staging for copy() — no more broken multi-stage builds ([bb1dfc4](https://github.com/AZX-PBC-OSS/containerspec/commit/bb1dfc46f98689e3a405356b952df220e85181ac))

## 0.1.0 (2026-07-22)


### Features

* containerspec v0.1.0 — fluent, content-hashed image builder ([a905a65](https://github.com/AZX-PBC-OSS/containerspec/commit/a905a6573ad43783b45e891b2c9d8137f416787f))


### Bug Fixes

* remove .serena from repo, add to .gitignore ([b57e3fc](https://github.com/AZX-PBC-OSS/containerspec/commit/b57e3fc4345e952f31237441d1328793e09128c7))

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
