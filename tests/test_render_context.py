from __future__ import annotations

from containerspec import ImageSpec


class TestRenderContextCacheMounts:
    def test_uv_cache_follows_user(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="warden")
            .uv_pip_install("httpx")
        )
        df = spec.to_dockerfile()
        assert "/home/warden/.cache/uv" in df
        assert "/root/.cache/uv" not in df

    def test_pip_cache_follows_user(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="warden")
            .pip_install("httpx")
        )
        df = spec.to_dockerfile()
        assert "/home/warden/.cache/pip" in df
        assert "/root/.cache/pip" not in df

    def test_npm_cache_follows_user(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .nvm_install("22")
            .user(uid=1000, gid=1000, name="warden")
            .npm_install("typescript")
        )
        df = spec.to_dockerfile()
        assert "/home/warden/.npm" in df
        assert "/root/.npm" not in df

    def test_cargo_cache_is_fixed_opt_path(self) -> None:
        """Cargo's cache mount targets CARGO_HOME=/opt/cargo regardless of user context.

        Rustup installs to a fixed /opt location (not $HOME) so cargo/rustc stay
        reachable after a later non-root .user() switch — /root defaults to 0700
        and is unreadable by anyone else. The cache mount must target the same
        path cargo actually reads from, so it can't be home-based either.
        """
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .rust_install()
            .user(uid=1000, gid=1000, name="warden")
            .cargo_install("ripgrep")
        )
        df = spec.to_dockerfile()
        assert "/opt/cargo/registry" in df
        assert "/home/warden/.cargo" not in df

    def test_uv_cache_root_when_no_user(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("httpx")
        df = spec.to_dockerfile()
        assert "/root/.cache/uv" in df


class TestRenderContextChownSandwich:
    def test_chown_after_user_has_root_sandwich(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="warden")
            .chown("/data")
        )
        df = spec.to_dockerfile()
        assert "USER root" in df
        assert "RUN mkdir -p /data && chown -R 1000:1000 /data" in df
        assert "USER 1000:1000" in df

    def test_chown_without_user_no_sandwich(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/data", uid=0, gid=0)
        df = spec.to_dockerfile()
        assert "USER root" not in df
        assert "RUN mkdir -p /data && chown -R 0:0 /data" in df

    def test_chown_before_user_no_sandwich(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .chown("/data", uid=1000, gid=1000)
            .user(uid=1000, gid=1000, name="warden")
        )
        df = spec.to_dockerfile()
        chown_section = df.split("# user(")[0]
        assert "USER root" not in chown_section


class TestRenderContextToolPaths:
    def test_nvm_env_uses_opt_when_installed_as_root(self) -> None:
        """NVM_DIR is /opt/nvm when nvm_install() runs before any .user() call.

        Regression: nvm previously always installed into ctx.home (/root when
        still root). /root defaults to 0700, so a later non-root USER can't
        traverse into it and every node/npm/npx symlink under /usr/local/bin
        126s. /opt/nvm is world-readable+executable (0755) by default, so it
        survives a later user switch.
        """
        spec = (
            ImageSpec.from_registry("base", pin_digest=False).apt_install("curl").nvm_install("22")
        )
        df = spec.to_dockerfile()
        assert "ENV NVM_DIR=/opt/nvm" in df
        assert "/root/.nvm" not in df

    def test_nvm_env_uses_own_home_when_installed_as_non_root(self) -> None:
        """NVM_DIR stays $HOME/.nvm when nvm_install() runs after .user() — that
        home is owned by the active user, so it's already safe and writable."""
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .apt_install("curl")
            .user(uid=1000, gid=1000, name="warden")
            .nvm_install("22")
        )
        df = spec.to_dockerfile()
        assert "ENV NVM_DIR=/home/warden/.nvm" in df
        assert "/opt/nvm" not in df

    def test_rust_path_uses_opt_when_installed_as_root(self) -> None:
        """CARGO_HOME/RUSTUP_HOME/PATH use /opt when rust_install() runs before .user().

        Same class of bug as nvm: rustup defaults to installing under $HOME
        (/root while still root), which becomes unreadable to a later
        non-root user.
        """
        spec = ImageSpec.from_registry("base", pin_digest=False).rust_install()
        df = spec.to_dockerfile()
        assert "ENV CARGO_HOME=/opt/cargo" in df
        assert "ENV RUSTUP_HOME=/opt/rustup" in df
        assert "ENV PATH=/opt/cargo/bin:$PATH" in df
        assert "/root/.cargo" not in df

    def test_rust_path_uses_own_home_when_installed_as_non_root(self) -> None:
        """CARGO_HOME/PATH stay $HOME/.cargo when rust_install() runs after .user()."""
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="warden")
            .rust_install()
        )
        df = spec.to_dockerfile()
        assert "ENV PATH=/home/warden/.cargo/bin:$PATH" in df
        assert "/opt/cargo" not in df

    def test_nvm_sets_npm_prefix(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).nvm_install("22")
        df = spec.to_dockerfile()
        assert "npm config set prefix /usr/local" in df

    def test_virtual_env_set(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).add_python("3.12")
        df = spec.to_dockerfile()
        assert "ENV VIRTUAL_ENV=/opt/venv" in df

    def test_pnpm_mount_on_run_line(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).pnpm_install("react")
        df = spec.to_dockerfile()
        # pnpm_install splits into a bootstrap RUN and an "add" RUN; the cache
        # mount lives on the "pnpm add" line (the pnpm store), not the bootstrap.
        pnpm_add_line = next(
            line for line in df.split("\n") if "pnpm" in line and "--mount=type=cache" in line
        )
        assert '"pnpm", "add"' in pnpm_add_line

    def test_brew_writes_shellenv_to_user_home(self) -> None:
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="warden")
            .brew_install("jq")
        )
        df = spec.to_dockerfile()
        assert "/home/warden/.bashrc" in df


class TestAlpineUserRendering:
    def test_user_alpine_explicit_flag(self) -> None:
        """user(alpine=True) renders addgroup/adduser even without preceding apk_install."""
        spec = ImageSpec.from_registry("alpine:3.20", pin_digest=False).user(
            uid=1000, gid=1000, name="warden", alpine=True
        )
        df = spec.to_dockerfile()
        assert "addgroup -g 1000" in df
        assert "adduser -u 1000" in df
        assert "groupadd" not in df
        assert "useradd" not in df

    def test_user_alpine_from_distro_param(self) -> None:
        """from_registry(distro='alpine') makes .user() render Alpine commands."""
        spec = ImageSpec.from_registry(
            "custom/alpine-based:latest", pin_digest=False, distro="alpine"
        ).user(uid=1000, gid=1000, name="warden")
        df = spec.to_dockerfile()
        assert "addgroup -g 1000" in df
        assert "adduser -u 1000" in df

    def test_user_debian_default(self) -> None:
        """Without alpine flag or distro='alpine', .user() renders Debian commands."""
        spec = ImageSpec.from_registry("base", pin_digest=False).user(
            uid=1000, gid=1000, name="warden"
        )
        df = spec.to_dockerfile()
        assert "groupadd -g 1000" in df
        assert "useradd -u 1000" in df

    def test_user_alpine_auto_detected_from_apk(self) -> None:
        """apk_install before user() auto-detects Alpine."""
        spec = (
            ImageSpec.from_registry("alpine:3.20", pin_digest=False)
            .apk_install("openrc")
            .user(uid=1000, gid=1000, name="warden")
        )
        df = spec.to_dockerfile()
        assert "addgroup" in df
        assert "adduser" in df
