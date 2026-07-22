"""Distro profiles — data-driven distro-specific command configuration.

Instead of branching in renderer functions, each distro declares its
user-creation commands, package-update commands, and non-interactive env.
Adding a new distro is a data entry, not code changes across renderers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistroProfile:
    """Distro-specific command templates for rendering."""

    group_add: str
    user_add: str
    package_update: str
    noninteractive_prefix: str


DISTRO_PROFILES: dict[str, DistroProfile] = {
    "debian": DistroProfile(
        group_add="groupadd -g {gid} {name}",
        user_add="useradd -u {uid} -g {gid} -m -d /home/{name} {name}",
        package_update=(
            "DEBIAN_FRONTEND=noninteractive apt-get update && "
            "apt-get dist-upgrade -y --no-install-recommends && "
            "apt-get clean && rm -rf /var/lib/apt/lists/*"
        ),
        noninteractive_prefix="DEBIAN_FRONTEND=noninteractive",
    ),
    "alpine": DistroProfile(
        group_add="addgroup -g {gid} {name}",
        user_add="adduser -u {uid} -G {name} -D -h /home/{name} {name}",
        package_update="apk update && apk upgrade --no-cache",
        noninteractive_prefix="",
    ),
    "rhel": DistroProfile(
        group_add="groupadd -g {gid} {name}",
        user_add="useradd -u {uid} -g {gid} -m -d /home/{name} {name}",
        package_update="dnf upgrade -y && dnf clean all",
        noninteractive_prefix="",
    ),
    "fedora": DistroProfile(
        group_add="groupadd -g {gid} {name}",
        user_add="useradd -u {uid} -g {gid} -m -d /home/{name} {name}",
        package_update="dnf upgrade -y && dnf clean all",
        noninteractive_prefix="",
    ),
    "arch": DistroProfile(
        group_add="groupadd -g {gid} {name}",
        user_add="useradd -u {uid} -g {gid} -m -d /home/{name} {name}",
        package_update="pacman -Syu --noconfirm",
        noninteractive_prefix="",
    ),
    "opensuse": DistroProfile(
        group_add="groupadd -g {gid} {name}",
        user_add="useradd -u {uid} -g {gid} -m -d /home/{name} {name}",
        package_update="zypper update -y",
        noninteractive_prefix="",
    ),
    "busybox": DistroProfile(
        group_add="addgroup -g {gid} {name}",
        user_add="adduser -u {uid} -G {name} -D -h /home/{name} {name}",
        package_update="apk update && apk upgrade --no-cache",
        noninteractive_prefix="",
    ),
}


def get_profile(distro: str | None) -> DistroProfile:
    """Get the distro profile, falling back to debian (the most common base)."""
    if distro and distro in DISTRO_PROFILES:
        return DISTRO_PROFILES[distro]
    return DISTRO_PROFILES["debian"]


def distro_from_pm(pm: str | None) -> str | None:
    """Infer distro from package manager used."""
    pm_to_distro = {
        "apt": "debian",
        "apk": "alpine",
        "dnf": "rhel",
        "pacman": "arch",
        "zypper": "opensuse",
    }
    return pm_to_distro.get(pm) if pm else None
