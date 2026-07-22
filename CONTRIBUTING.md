# Contributing to ContainerSpec

Thanks for your interest in contributing! This guide covers the basics of
getting set up and landing a change.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (the project's package manager)

## Setup

```bash
git clone https://github.com/AZX-PBC-OSS/containerspec.git
cd containerspec
uv sync --group dev
```

## Development commands

The [`Makefile`](Makefile) exposes the common workflows:

```bash
make lint        # ruff check
make format      # ruff format + auto-fix
make type-check  # pyright on src/
make test        # pytest
```

Before opening a PR, make sure all of these pass:

```bash
make lint
make type-check
make test
```

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/).
Prefix each commit with a type so the changelog is generated automatically:

```
feat: add support for podman builds
fix: correct layer ordering in OCI export
docs: clarify content-hash behavior in README
```

Common types: `feat`, `fix`, `perf`, `refactor`, `docs`, `build`, `ci`,
`revert`. Style, chore, and test commits are hidden from the changelog.

## Pull request process

1. Fork the repository and create a branch from `main`.
2. Make your changes, keeping files small and focused.
3. Run `make lint`, `make type-check`, and `make test` locally.
4. Open a pull request against `main` using the PR template.
5. Ensure CI passes on all matrix jobs (lint, typecheck, test, security, build).

## Releases

Releases are automated via [release-please](https://github.com/googleapis/release-please).
Merging a release PR to `main` creates a tag and triggers publication to PyPI
through the centralized reusable workflow. No manual publish step is required.
