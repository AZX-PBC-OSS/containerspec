"""Dockerfile rendering with build context state tracking.

Dockerfile is a stateful, sequential format — USER, WORKDIR, ENV persist
for all subsequent layers. The RenderContext tracks this accumulated state
so each layer renders with correct paths, cache mounts, and user switches.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from containerspec.distros import distro_from_pm, get_profile
from containerspec.layers import (
    AddPython,
    ApkInstall,
    AptInstall,
    BrewInstall,
    CargoInstall,
    Chown,
    Cmd,
    Copy,
    CopyFromStage,
    DnfInstall,
    Entrypoint,
    Env,
    Expose,
    GemInstall,
    GoInstall,
    Layer,
    NpmInstall,
    NvmInstall,
    PacmanInstall,
    PipInstall,
    PnpmInstall,
    RunCommands,
    RustInstall,
    User,
    UvPipInstall,
    UvxInstall,
    Volume,
    Workdir,
    YarnInstall,
    ZypperInstall,
)

if TYPE_CHECKING:
    from containerspec.spec import ImageSpec

NVM_VERSION = "0.40.6"
NVM_INSTALL_SCRIPT = f"https://raw.githubusercontent.com/nvm-sh/nvm/v{NVM_VERSION}/install.sh"

# Defense-in-depth: rejects shell metacharacters (;, |, &, $, backtick, parens,
# braces, spaces, quotes, glob chars) that would be dangerous if ever rendered
# into shell-form RUN. PEP 508 / npm semver characters (>=, <, >, !, ~, ,, ^, #)
# and a leading @ (npm scoped packages) are allowed because install renderers
# emit exec-form RUN, which bypasses the shell entirely — arguments are passed
# as literal JSON-array strings with no shell interpretation.
_PKG_PATTERN = re.compile(r"^@?[A-Za-z0-9][A-Za-z0-9._+:=@~/\[\]<>!~,^#-]*$")


def validate_package(name: str) -> str:
    """Validate a package name against a safe pattern to prevent shell injection."""
    if not name:
        raise ValueError("Package name cannot be empty")
    if not _PKG_PATTERN.match(name):
        raise ValueError(
            f"Invalid package name: {name!r}. Package names must match {_PKG_PATTERN.pattern}"
        )
    return name


def _exec_run(mounts: list[str], args: list[str]) -> str:
    """Render an exec-form RUN line with optional BuildKit cache ``mounts``.

    Exec form (``RUN ["cmd", "arg"]``) bypasses the shell completely — the JSON
    array elements are passed to ``exec`` as literal strings, so no shell
    quoting is needed and version specifiers like ``httpx>=0.27.0`` are safe.
    ``mounts`` are ``--mount=type=cache,...`` flags that BuildKit parses before
    the exec array.
    """
    return " ".join(["RUN", *mounts, json.dumps(args)])


@dataclass
class RenderContext:
    """Tracks accumulated Dockerfile state from preceding layers."""

    current_user: User | None = None
    current_workdir: str = "/"
    python_venv: str | None = None
    rust_installed: bool = False
    nvm_installed: bool = False
    brew_installed: bool = False
    package_manager: str | None = None
    distro: str | None = None

    @property
    def home(self) -> str:
        if self.current_user and self.current_user.uid != 0:
            return f"/home/{self.current_user.name}"
        return "/root"

    @property
    def is_root(self) -> bool:
        return self.current_user is None or self.current_user.uid == 0

    @property
    def uid(self) -> int:
        return self.current_user.uid if self.current_user else 0

    @property
    def gid(self) -> int:
        return self.current_user.gid if self.current_user else 0

    @property
    def is_alpine(self) -> bool:
        return self.package_manager == "apk"

    def uv_cache(self) -> str:
        return f"{self.home}/.cache/uv"

    def pip_cache(self) -> str:
        return f"{self.home}/.cache/pip"

    def npm_cache(self) -> str:
        return f"{self.home}/.npm"

    def cargo_registry(self) -> str:
        return f"{self.home}/.cargo/registry"

    def cargo_git(self) -> str:
        return f"{self.home}/.cargo/git"

    def pnpm_store(self) -> str:
        return f"{self.home}/.local/share/pnpm/store"

    def update(self, layer: Layer) -> None:
        """Update context state after a layer has been rendered."""
        if isinstance(layer, User):
            self.current_user = layer
        elif isinstance(layer, Workdir):
            self.current_workdir = layer.path
        elif isinstance(layer, AddPython):
            self.python_venv = "/opt/venv"
        elif isinstance(layer, RustInstall):
            self.rust_installed = True
        elif isinstance(layer, NvmInstall):
            self.nvm_installed = True
        elif isinstance(layer, BrewInstall):
            self.brew_installed = True
        elif isinstance(layer, AptInstall):
            self.package_manager = "apt"
        elif isinstance(layer, ApkInstall):
            self.package_manager = "apk"
        elif isinstance(layer, DnfInstall):
            self.package_manager = "dnf"
        elif isinstance(layer, PacmanInstall):
            self.package_manager = "pacman"
        elif isinstance(layer, ZypperInstall):
            self.package_manager = "zypper"


_DISTRO_TO_PM = {
    "debian": "apt",
    "alpine": "apk",
    "rhel": "dnf",
    "fedora": "dnf",
    "arch": "pacman",
    "opensuse": "zypper",
    "busybox": "apk",
}


def render_dockerfile(spec: ImageSpec, layer: Layer, index: int) -> list[str]:
    """Render a single layer with context from preceding layers."""
    ctx = RenderContext()
    if spec.distro:
        ctx.distro = spec.distro
        if spec.distro in _DISTRO_TO_PM:
            ctx.package_manager = _DISTRO_TO_PM[spec.distro]
    for preceding in spec.layers[:index]:
        ctx.update(preceding)
    return render_layer(spec, layer, index, ctx)


def render_layer(spec: ImageSpec, layer: Layer, index: int, ctx: RenderContext) -> list[str]:
    """Render a single layer to Dockerfile lines using the current build context."""
    if isinstance(layer, AddPython):
        return _render_add_python(layer)
    if isinstance(layer, AptInstall):
        return _render_apt_install(layer)
    if isinstance(layer, ApkInstall):
        return _render_apk_install(layer)
    if isinstance(layer, DnfInstall):
        return _render_dnf_install(layer)
    if isinstance(layer, PacmanInstall):
        return _render_pacman_install(layer)
    if isinstance(layer, ZypperInstall):
        return _render_zypper_install(layer)
    if isinstance(layer, UvPipInstall):
        return _render_uv_pip_install(layer, ctx)
    if isinstance(layer, PipInstall):
        return _render_pip_install(layer, ctx)
    if isinstance(layer, Env):
        return _render_env(layer)
    if isinstance(layer, RunCommands):
        return _render_run_commands(layer)
    if isinstance(layer, Workdir):
        return _render_workdir(layer)
    if isinstance(layer, Chown):
        return _render_chown(spec, layer, index, ctx)
    if isinstance(layer, User):
        return _render_user(layer, ctx)
    if isinstance(layer, Entrypoint):
        return _render_entrypoint(layer)
    if isinstance(layer, Expose):
        return _render_expose(layer)
    if isinstance(layer, Cmd):
        return _render_cmd(layer)
    if isinstance(layer, Volume):
        return _render_volume(layer)
    if isinstance(layer, Copy):
        return _render_copy(layer)
    if isinstance(layer, CopyFromStage):
        return _render_copy_from_stage(layer)
    if isinstance(layer, NvmInstall):
        return _render_nvm_install(layer, ctx)
    if isinstance(layer, NpmInstall):
        return _render_npm_install(layer, ctx)
    if isinstance(layer, PnpmInstall):
        return _render_pnpm_install(layer, ctx)
    if isinstance(layer, BrewInstall):
        return _render_brew_install(layer, ctx)
    if isinstance(layer, RustInstall):
        return _render_rust_install(ctx)
    if isinstance(layer, CargoInstall):
        return _render_cargo_install(layer, ctx)
    if isinstance(layer, UvxInstall):
        return _render_uvx_install(layer, ctx)
    if isinstance(layer, YarnInstall):
        return _render_yarn_install(layer, ctx)
    if isinstance(layer, GemInstall):
        return _render_gem_install(layer, ctx)
    if isinstance(layer, GoInstall):
        return _render_go_install(layer, ctx)
    raise TypeError(f"Unknown layer type: {type(layer).__name__}")


def _render_add_python(layer: AddPython) -> list[str]:
    return [
        f'# add_python("{layer.version}")',
        "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/",
        f"RUN uv python install {layer.version} && uv venv --python {layer.version} /opt/venv",
        "ENV PATH=/opt/venv/bin:$PATH",
        "ENV VIRTUAL_ENV=/opt/venv",
    ]


def _render_apt_install(layer: AptInstall) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    mounts = [
        "--mount=type=cache,target=/var/cache/apt,sharing=locked",
        "--mount=type=cache,target=/var/lib/apt,sharing=locked",
    ]
    # Split update/install into separate exec-form RUN lines so no shell is
    # invoked. DEBIAN_FRONTEND persists via ENV for all subsequent layers.
    # Both mounts appear on each line: /var/lib/apt is a shared BuildKit cache,
    # so update writes the apt lists into it and install reads from the same
    # cache (mounting it only on install would shadow the fresh lists).
    return [
        f"# apt_install({pkgs_repr})",
        "ENV DEBIAN_FRONTEND=noninteractive",
        _exec_run(mounts, ["apt-get", "update"]),
        _exec_run(mounts, ["apt-get", "install", "-y", "--no-install-recommends", *validated]),
    ]


def _render_apk_install(layer: ApkInstall) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    return [
        f"# apk_install({pkgs_repr})",
        _exec_run(
            ["--mount=type=cache,target=/var/cache/apk,sharing=locked"],
            ["apk", "add", "--no-cache", *validated],
        ),
    ]


def _render_dnf_install(layer: DnfInstall) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    return [
        f"# dnf_install({pkgs_repr})",
        _exec_run(
            ["--mount=type=cache,target=/var/cache/dnf,sharing=locked"],
            ["dnf", "install", "-y", *validated],
        ),
        _exec_run([], ["dnf", "clean", "all"]),
    ]


def _render_uv_pip_install(layer: UvPipInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    cache = ctx.uv_cache()
    # UV_LINK_MODE persists via ENV for subsequent layers (standard pattern).
    return [
        f"# uv_pip_install({pkgs_repr})",
        "ENV UV_LINK_MODE=copy",
        _exec_run(
            [f"--mount=type=cache,target={cache},sharing=locked"],
            ["uv", "pip", "install", *validated],
        ),
    ]


def _render_pip_install(layer: PipInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    cache = ctx.pip_cache()
    return [
        f"# pip_install({pkgs_repr})",
        _exec_run(
            [f"--mount=type=cache,target={cache},sharing=locked"],
            ["pip", "install", "--no-cache-dir", *validated],
        ),
    ]


def _render_env(layer: Env) -> list[str]:
    lines = [f"# env({json.dumps(dict(layer.vars))})"]
    for k, v in layer.vars.items():
        if " " in v or '"' in v:
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'ENV {k}="{escaped}"')
        else:
            lines.append(f"ENV {k}={v}")
    return lines


def _render_run_commands(layer: RunCommands) -> list[str]:
    cmds_repr = ", ".join(f'"{c}"' for c in layer.commands)
    if len(layer.commands) == 1:
        return [f"# run_commands({cmds_repr})", f"RUN {layer.commands[0]}"]
    chained = " && ".join(layer.commands)
    return [f"# run_commands({cmds_repr})", f"RUN {chained}"]


def _render_workdir(layer: Workdir) -> list[str]:
    return [f'# workdir("{layer.path}")', f"WORKDIR {layer.path}"]


def _render_chown(spec: ImageSpec, layer: Chown, index: int, ctx: RenderContext) -> list[str]:
    """Render chown with USER root sandwich when non-root user is active."""
    uid, gid = spec.resolve_chown_uid_gid(layer, index=index)
    path = layer.path
    source = " from preceding .user()" if layer.uid is None and layer.gid is None else ""
    lines: list[str] = [f'# chown("{path}") — resolved to uid={uid}, gid={gid}{source}']

    if not ctx.is_root:
        lines.append("USER root")
        lines.append(f"RUN mkdir -p {path} && chown -R {uid}:{gid} {path}")
        lines.append(f"USER {ctx.uid}:{ctx.gid}")
    else:
        lines.append(f"RUN mkdir -p {path} && chown -R {uid}:{gid} {path}")
    return lines


def _render_user(layer: User, ctx: RenderContext) -> list[str]:
    """Render user creation using distro-specific commands from DistroProfile."""
    effective_distro = (
        (layer.alpine and "alpine")
        or (ctx.package_manager and distro_from_pm(ctx.package_manager))
        or ctx.distro
    )
    profile = get_profile(effective_distro if effective_distro else "debian")
    group_cmd = profile.group_add.format(gid=layer.gid, name=layer.name)
    user_cmd = profile.user_add.format(uid=layer.uid, gid=layer.gid, name=layer.name)
    return [
        f'# user(uid={layer.uid}, gid={layer.gid}, name="{layer.name}")',
        f"RUN {group_cmd} && {user_cmd}",
        f"USER {layer.uid}:{layer.gid}",
    ]


def _render_entrypoint(layer: Entrypoint) -> list[str]:
    if layer.commands is None:
        return ["# entrypoint(None)"]
    cmds_repr = ", ".join(f'"{c}"' for c in layer.commands)
    return [f"# entrypoint([{cmds_repr}])", f"ENTRYPOINT {json.dumps(list(layer.commands))}"]


def _render_expose(layer: Expose) -> list[str]:
    ports_repr = ", ".join(str(p) for p in layer.ports)
    return [f"# expose({ports_repr})", f"EXPOSE {' '.join(str(p) for p in layer.ports)}"]


def _render_cmd(layer: Cmd) -> list[str]:
    if layer.commands is None:
        return ["# cmd(None)"]
    cmds_repr = ", ".join(f'"{c}"' for c in layer.commands)
    return [f"# cmd([{cmds_repr}])", f"CMD {json.dumps(list(layer.commands))}"]


def _render_volume(layer: Volume) -> list[str]:
    paths_repr = ", ".join(f'"{p}"' for p in layer.paths)
    return [f"# volume({paths_repr})", f"VOLUME {json.dumps(list(layer.paths))}"]


def _render_copy(layer: Copy) -> list[str]:
    return [
        f'# copy("{layer.src}", "{layer.dest}")',
        f"COPY {layer.src} {layer.dest}",
    ]


def _render_copy_from_stage(layer: CopyFromStage) -> list[str]:
    return [
        f'# copy_from_stage("{layer.stage_name}", "{layer.src}", "{layer.dest}")',
        f"COPY --from={layer.stage_name} {layer.src} {layer.dest}",
    ]


def _render_nvm_install(layer: NvmInstall, ctx: RenderContext) -> list[str]:
    version = layer.version
    nvm_dir = f"{ctx.home}/.nvm"
    return [
        f'# nvm_install("{version}")',
        'SHELL ["/bin/bash", "-o", "pipefail", "-c"]',
        f"RUN curl -o- {NVM_INSTALL_SCRIPT} | bash \\\n"
        f'    && export NVM_DIR="{nvm_dir}" && . "$NVM_DIR/nvm.sh" \\\n'
        f"    && nvm install {version} \\\n"
        f"    && nvm alias default {version} \\\n"
        f"    && npm config set prefix /usr/local \\\n"
        f"    && NODE_DIR=$(dirname $(which node)) \\\n"
        f"    && ln -sf $NODE_DIR/node /usr/local/bin/node \\\n"
        f"    && ln -sf $NODE_DIR/npm /usr/local/bin/npm \\\n"
        f"    && ln -sf $NODE_DIR/npx /usr/local/bin/npx",
        f"ENV NVM_DIR={nvm_dir}",
    ]


def _render_npm_install(layer: NpmInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    cache = ctx.npm_cache()
    return [
        f"# npm_install({pkgs_repr})",
        _exec_run(
            [f"--mount=type=cache,target={cache},sharing=locked"],
            ["npm", "install", "-g", *validated],
        ),
    ]


def _render_pnpm_install(layer: PnpmInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    store = ctx.pnpm_store()
    # Split: install pnpm itself, then add packages with the pnpm store cache.
    return [
        f"# pnpm_install({pkgs_repr})",
        _exec_run([], ["npm", "install", "-g", "pnpm"]),
        _exec_run(
            [f"--mount=type=cache,target={store},sharing=locked"],
            ["pnpm", "add", "-g", *validated],
        ),
    ]


def _render_brew_install(layer: BrewInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    brew_prefix = "/home/linuxbrew/.linuxbrew"
    # The Homebrew setup script is the library's own code (no user input), so it
    # stays shell-form. The package install is exec-form: ENV PATH puts brew on
    # PATH for the subsequent exec-form RUN, so no shell is needed to find it.
    return [
        f"# brew_install({pkgs_repr})",
        'SHELL ["/bin/bash", "-o", "pipefail", "-c"]',
        "RUN if ! command -v brew &>/dev/null; then \\\n"
        '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \\\n'
        f"    echo 'eval \"$({brew_prefix}/bin/brew shellenv)\"' >> {ctx.home}/.bashrc; \\\n"
        "fi",
        f"ENV PATH={brew_prefix}/bin:$PATH",
        _exec_run([], ["brew", "install", *validated]),
    ]


def _render_rust_install(ctx: RenderContext) -> list[str]:
    cargo_bin = f"{ctx.home}/.cargo/bin"
    return [
        "# rust_install()",
        'SHELL ["/bin/bash", "-o", "pipefail", "-c"]',
        "RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        f"ENV PATH={cargo_bin}:$PATH",
    ]


def _render_cargo_install(layer: CargoInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    registry = ctx.cargo_registry()
    git = ctx.cargo_git()
    return [
        f"# cargo_install({pkgs_repr})",
        _exec_run(
            [
                f"--mount=type=cache,target={registry},sharing=locked",
                f"--mount=type=cache,target={git},sharing=locked",
            ],
            ["cargo", "install", *validated],
        ),
    ]


def _render_uvx_install(layer: UvxInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    cache = ctx.uv_cache()
    return [
        f"# uvx_install({pkgs_repr})",
        _exec_run(
            [f"--mount=type=cache,target={cache},sharing=locked"],
            ["uvx", "--system", *validated],
        ),
    ]


def _render_pacman_install(layer: PacmanInstall) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    return [
        f"# pacman_install({pkgs_repr})",
        _exec_run([], ["pacman", "-S", "--noconfirm", "--needed", *validated]),
        _exec_run([], ["pacman", "-Scc", "--noconfirm"]),
    ]


def _render_zypper_install(layer: ZypperInstall) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    return [
        f"# zypper_install({pkgs_repr})",
        _exec_run([], ["zypper", "install", "-y", *validated]),
        _exec_run([], ["zypper", "clean", "-a"]),
    ]


def _render_yarn_install(layer: YarnInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    cache = f"{ctx.home}/.cache/yarn"
    return [
        f"# yarn_install({pkgs_repr})",
        _exec_run(
            [f"--mount=type=cache,target={cache},sharing=locked"],
            ["yarn", "global", "add", *validated],
        ),
    ]


def _render_gem_install(layer: GemInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    return [
        f"# gem_install({pkgs_repr})",
        _exec_run([], ["gem", "install", *validated]),
    ]


def _render_go_install(layer: GoInstall, ctx: RenderContext) -> list[str]:
    pkgs_repr = ", ".join(f'"{p}"' for p in layer.packages)
    validated = [validate_package(p) for p in layer.packages]
    cache = f"{ctx.home}/.cache/go-build"
    return [
        f"# go_install({pkgs_repr})",
        _exec_run(
            [f"--mount=type=cache,target={cache},sharing=locked"],
            ["go", "install", *validated],
        ),
    ]
