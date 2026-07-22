"""Scenario tests that do not require Docker.

Covers hash stability and pure Dockerfile generation. Runs in default CI
without CONTAINERSPEC_E2E=1 and without a Docker daemon.
"""

from __future__ import annotations

from containerspec import ImageSpec


class TestHashStability:
    def test_reordered_packages_same_hash(self) -> None:
        """apt_install order doesn't change the hash."""
        a = ImageSpec.from_registry("base", pin_digest=False).apt_install("git", "curl", "jq")
        b = ImageSpec.from_registry("base", pin_digest=False).apt_install("jq", "git", "curl")
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_different_package_managers_different_hash(self) -> None:
        """apt_install vs apk_install vs dnf_install produce different hashes."""
        apt = ImageSpec.from_registry("base", pin_digest=False).apt_install("git")
        apk = ImageSpec.from_registry("base", pin_digest=False).apk_install("git")
        dnf = ImageSpec.from_registry("base", pin_digest=False).dnf_install("git")
        assert apt.content_hash(client=None) != apk.content_hash(client=None)
        assert apt.content_hash(client=None) != dnf.content_hash(client=None)
        assert apk.content_hash(client=None) != dnf.content_hash(client=None)


class TestPureDockerfileGeneration:
    def test_dockerfile_no_docker_needed(self, tmp_path) -> None:
        """to_dockerfile() works without Docker installed."""
        app_py = tmp_path / "app.py"
        app_py.write_text("print('hello')")
        spec = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .add_python("3.12")
            .apt_install("git", "curl")
            .uv_pip_install("httpx", "rich")
            .pip_install("requests")
            .env({"ENV1": "val1", "ENV2": "val2"})
            .workdir("/app")
            .run_commands("echo hello", "echo world")
            .user(uid=1000, gid=1000, name="dev")
            .chown("/app")
            .expose(8000, 8080)
            .cmd(["python", "main.py"])
            .volume("/data", "/cache")
            .copy(str(app_py), "/app/app.py")
            .entrypoint(["python"])
        )
        df = spec.to_dockerfile()
        assert "# syntax=docker/dockerfile:1.7" in df
        assert "FROM python:3.12-slim" in df
        assert "ENV VIRTUAL_ENV=/opt/venv" in df
        # Install renderers use exec form (JSON arrays), so the command tokens
        # appear as JSON-quoted strings rather than space-separated shell words.
        assert '"apt-get", "install"' in df
        assert '"uv", "pip", "install"' in df
        assert '"pip", "install"' in df
        assert "WORKDIR /app" in df
        assert "USER root" in df
        assert "USER 1000:1000" in df
        assert "EXPOSE 8000 8080" in df
        assert "CMD [" in df
        assert "VOLUME [" in df
        assert f"COPY {app_py} /app/app.py" in df
        assert "ENTRYPOINT [" in df
        h = spec.content_hash(client=None)
        assert len(h) == 64
        tag = spec.tag("myimage", client=None)
        assert tag.startswith("myimage:sha-")
