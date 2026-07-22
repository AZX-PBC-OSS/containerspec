"""Allowlist validators for user-controlled fields that reach the Dockerfile.

Package names, user/stage names, tool versions, filesystem paths, env vars, and
image references are interpolated into shell-form ``RUN`` lines and Dockerfile
directives, so a spec built from untrusted input could otherwise inject commands
or directives at build time. Each field kind has a narrow allowlist; unsafe
input is rejected with a clear ``ValueError`` rather than silently quoted.
"""

from __future__ import annotations

import re

# Defense-in-depth: rejects shell metacharacters (;, |, &, $, backtick, parens,
# braces, spaces, quotes, glob chars) that would be dangerous if ever rendered
# into shell-form RUN. PEP 508 / npm semver characters (>=, <, >, !, ~, ,, ^, #)
# and a leading @ (npm scoped packages) are allowed because install renderers
# emit exec-form RUN, which bypasses the shell entirely — arguments are passed
# as literal JSON-array strings with no shell interpretation.
_PKG_PATTERN = re.compile(r"^@?[A-Za-z0-9][A-Za-z0-9._+:=@~/\[\]<>!~,^#-]*$")

# The remaining fields render into shell-form RUN lines and COPY/WORKDIR/FROM/ENV
# directives, so they must never carry shell metacharacters, whitespace, or
# newlines (which would break out of a command or inject a new directive).
# fullmatch anchors implicitly and — unlike a trailing ``$`` — does not accept a
# trailing newline. Allowlists by field kind:
#   name     — user/stage names: strict, no slash/glob.
#   version  — tool versions: also allow ``/`` and ``@`` (nvm ``lts/hydrogen``,
#              uv ``pypy@3.10``); glob ``*`` stays out (dangerous in shell RUN).
#   path     — COPY paths: also allow leading ``.``, ``..``, and glob ``*``
#              (``COPY . /app``, ``.env``, ``src/*.py``) — COPY is not shell.
#   fs_path  — chown/workdir paths: NO glob/tilde, since those render into a
#              shell-form RUN (or a directive) where the shell would expand them.
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*")
_PATH_PATTERN = re.compile(r"[A-Za-z0-9._/~*][A-Za-z0-9._/~*+-]*")
_FS_PATH_PATTERN = re.compile(r"[A-Za-z0-9._/][A-Za-z0-9._/+-]*")
# Env var names: allow a leading underscore (``_JAVA_OPTIONS``).
_ENV_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Image references (FROM): registry/repo:tag@digest — allow ``:`` and ``@``.
_IMAGE_REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*")


def validate_package(name: str) -> str:
    """Validate a package name against a safe pattern to prevent shell injection."""
    if not name:
        raise ValueError("Package name cannot be empty")
    if not _PKG_PATTERN.match(name):
        raise ValueError(
            f"Invalid package name: {name!r}. Package names must match {_PKG_PATTERN.pattern}"
        )
    return name


def _validate(value: str, *, field: str, pattern: re.Pattern[str]) -> str:
    if not value:
        raise ValueError(f"{field} cannot be empty")
    if not pattern.fullmatch(value):
        raise ValueError(
            f"Invalid {field}: {value!r}. Must match {pattern.pattern} "
            f"— shell metacharacters, whitespace, and newlines are not allowed"
        )
    return value


def validate_name(value: str, *, field: str) -> str:
    """Validate a user/stage name against shell/directive injection."""
    return _validate(value, field=field, pattern=_NAME_PATTERN)


def validate_version(value: str, *, field: str) -> str:
    """Validate a tool version selector against shell/directive injection."""
    return _validate(value, field=field, pattern=_VERSION_PATTERN)


def validate_path(value: str, *, field: str) -> str:
    """Validate a COPY path (globbing allowed) against shell/directive injection."""
    return _validate(value, field=field, pattern=_PATH_PATTERN)


def validate_fs_path(value: str, *, field: str) -> str:
    """Validate a chown/workdir path — no glob/tilde (renders into shell form)."""
    return _validate(value, field=field, pattern=_FS_PATH_PATTERN)


def validate_env_key(value: str, *, field: str) -> str:
    """Validate an environment variable name (leading underscore allowed)."""
    return _validate(value, field=field, pattern=_ENV_KEY_PATTERN)


def validate_image_ref(value: str, *, field: str) -> str:
    """Validate a FROM image reference against directive injection."""
    return _validate(value, field=field, pattern=_IMAGE_REF_PATTERN)
