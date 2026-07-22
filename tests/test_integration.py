"""Real integration tests. Gated behind CONTAINERSPEC_E2E=1.

Uses real Docker + buildx to build images, rootfs, and OCI tarballs.
Tests that containers actually run and tools work end-to-end.
Not mocked — these verify the real build pipeline.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _require_docker() -> None:
    if os.environ.get("CONTAINERSPEC_E2E") != "1":
        pytest.skip("set CONTAINERSPEC_E2E=1 to run integration tests")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:  # noqa: S607
        pytest.skip("docker daemon not reachable")


def _docker_run(tag: str, *cmd: str, timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(  # noqa: S603
        ["docker", "run", "--rm", tag, *cmd],  # noqa: S607
        capture_output=True,
        timeout=timeout,
        check=False,
    )


class TestDockerTargetBuild:
    @pytest.mark.asyncio
    async def test_python_pip_build_and_run(self) -> None:
        from containerspec import ImageSpec

        spec = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("httpx")
            .entrypoint([])
        )
        built = await spec.build("containerspec-test-pip")
        assert built.tag.startswith("containerspec-test-pip:sha-")

        result = _docker_run(built.tag, "python", "-c", "import httpx; print('ok')")
        assert result.returncode == 0, result.stderr.decode()
        assert b"ok" in result.stdout

        built_again = await spec.build("containerspec-test-pip")
        assert built_again.tag == built.tag

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607

    @pytest.mark.asyncio
    async def test_python_uv_pip_build_and_run(self) -> None:
        from containerspec import ImageSpec

        spec = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .add_python("3.12")
            .uv_pip_install("httpx")
            .entrypoint([])
        )
        built = await spec.build("containerspec-test-uv")
        assert built.tag.startswith("containerspec-test-uv:sha-")

        result = _docker_run(built.tag, "python", "-c", "import httpx; print('uv-ok')")
        assert result.returncode == 0, result.stderr.decode()
        assert b"uv-ok" in result.stdout

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607

    @pytest.mark.asyncio
    async def test_node_nvm_build_and_run(self) -> None:
        from containerspec import ImageSpec

        spec = (
            ImageSpec.from_registry("debian:bookworm-slim", pin_digest=False)
            .apt_install("curl", "ca-certificates")
            .nvm_install("22")
            .npm_install("typescript")
            .entrypoint([])
        )
        built = await spec.build("containerspec-test-nvm")
        assert built.tag.startswith("containerspec-test-nvm:sha-")

        result = _docker_run(built.tag, "node", "--version", timeout=120)
        assert result.returncode == 0, result.stderr.decode()
        assert b"v22" in result.stdout

        result_ts = _docker_run(built.tag, "tsc", "--version", timeout=60)
        assert result_ts.returncode == 0, result_ts.stderr.decode()

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607

    @pytest.mark.asyncio
    async def test_rust_cargo_build_and_run(self) -> None:
        from containerspec import ImageSpec

        spec = (
            ImageSpec.from_registry("debian:bookworm-slim", pin_digest=False)
            .apt_install("curl", "ca-certificates", "build-essential")
            .rust_install()
            .cargo_install("ripgrep")
            .entrypoint([])
        )
        built = await spec.build("containerspec-test-rust")
        assert built.tag.startswith("containerspec-test-rust:sha-")

        result = _docker_run(built.tag, "rg", "--version", timeout=180)
        assert result.returncode == 0, result.stderr.decode()
        assert b"ripgrep" in result.stdout

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607

    @pytest.mark.asyncio
    async def test_expose_and_cmd(self) -> None:
        from containerspec import ImageSpec

        spec = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("httpx")
            .expose(8000)
            .cmd(["python", "-c", "print('cmd-works')"])
            .entrypoint([])
        )
        built = await spec.build("containerspec-test-cmd")
        result = _docker_run(built.tag, timeout=30)
        assert result.returncode == 0, result.stderr.decode()
        assert b"cmd-works" in result.stdout

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607


class TestOciTargetBuild:
    @pytest.mark.asyncio
    async def test_oci_tarball_builds(self, tmp_path: Path) -> None:
        from containerspec import ImageSpec, OciTarget

        spec = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("httpx")
            .entrypoint([])
        )
        oci_path = tmp_path / "image.tar"
        result = await spec.build(OciTarget(path=str(oci_path)))
        assert oci_path.exists()
        assert result.path == str(oci_path)
        assert (tmp_path / "image.tar.containerspec.json").exists()


class TestFirecrackerRootfsBuild:
    @pytest.mark.asyncio
    async def test_rootfs_alpine_builds(self, tmp_path: Path) -> None:
        from containerspec import FirecrackerRootfsTarget, ImageSpec

        spec = (
            ImageSpec.from_registry("alpine:3.20", pin_digest=False)
            .apk_install("openrc")
            .entrypoint([])
        )
        rootfs_path = tmp_path / "rootfs.ext4"
        result = await spec.build(
            FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=256, converter="mke2fs")
        )
        assert rootfs_path.exists()
        assert result.path == str(rootfs_path)
        assert (tmp_path / "rootfs.ext4.containerspec.json").exists()

    @pytest.mark.asyncio
    async def test_rootfs_second_build_skips(self, tmp_path: Path) -> None:
        from containerspec import FirecrackerRootfsTarget, ImageSpec

        spec = (
            ImageSpec.from_registry("alpine:3.20", pin_digest=False)
            .apk_install("openrc")
            .entrypoint([])
        )
        rootfs_path = tmp_path / "rootfs.ext4"
        target = FirecrackerRootfsTarget(path=str(rootfs_path), size_mb=256, converter="mke2fs")
        first = await spec.build(target)
        second = await spec.build(target)
        assert first.hash == second.hash
        assert first.path == second.path


class TestConcurrentBuilds:
    @pytest.mark.asyncio
    async def test_concurrent_different_specs(self) -> None:
        from containerspec import ImageSpec

        spec_a = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("httpx")
            .entrypoint([])
        )
        spec_b = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("rich")
            .entrypoint([])
        )

        async with asyncio.TaskGroup() as tg:
            task_a = tg.create_task(spec_a.build("containerspec-test-conc-a"))
            task_b = tg.create_task(spec_b.build("containerspec-test-conc-b"))

        built_a = task_a.result()
        built_b = task_b.result()
        assert built_a.tag != built_b.tag

        result_a = _docker_run(built_a.tag, "python", "-c", "import httpx; print('a')")
        assert result_a.returncode == 0, result_a.stderr.decode()
        result_b = _docker_run(built_b.tag, "python", "-c", "import rich; print('b')")
        assert result_b.returncode == 0, result_b.stderr.decode()

        subprocess.run(["docker", "rmi", built_a.tag, built_b.tag], check=False, timeout=30)  # noqa: S603, S607


class TestRootlessBuild:
    @pytest.mark.asyncio
    async def test_rootless_python_build_runs_as_user(self) -> None:
        """Build a rootless image: user 1000, chown HF_HOME, verify `whoami` returns the user."""
        from containerspec import ImageSpec

        spec = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("httpx")
            .env({"HF_HOME": "/home/warden/.cache/huggingface"})
            .user(uid=1000, gid=1000, name="warden")
            .chown("/home/warden/.cache/huggingface")
            .entrypoint([])
        )
        built = await spec.build("containerspec-test-rootless")

        result = _docker_run(built.tag, "whoami", timeout=30)
        assert result.returncode == 0, result.stderr.decode()
        assert b"warden" in result.stdout

        result2 = _docker_run(built.tag, "python", "-c", "import httpx; print('ok')", timeout=30)
        assert result2.returncode == 0, result2.stderr.decode()

        result3 = _docker_run(
            built.tag,
            "python",
            "-c",
            "import os; print(os.environ.get('HF_HOME')); os.makedirs(os.environ['HF_HOME'], exist_ok=True); print('writable')",
            timeout=30,
        )
        assert result3.returncode == 0, result3.stderr.decode()
        assert b"/home/warden/.cache/huggingface" in result3.stdout
        assert b"writable" in result3.stdout

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607


class TestAlpineBuild:
    @pytest.mark.asyncio
    async def test_alpine_apk_build_and_run(self) -> None:
        """Build an Alpine image with apk_install, verify packages are present."""
        from containerspec import ImageSpec

        spec = (
            ImageSpec.from_registry("alpine:3.20", pin_digest=False)
            .apk_install("jq", "curl")
            .entrypoint([])
        )
        built = await spec.build("containerspec-test-alpine")

        result = _docker_run(built.tag, "jq", "--version", timeout=30)
        assert result.returncode == 0, result.stderr.decode()
        assert b"jq" in result.stdout

        result2 = _docker_run(built.tag, "curl", "--version", timeout=30)
        assert result2.returncode == 0, result2.stderr.decode()

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607


class TestCacheConsistency:
    @pytest.mark.asyncio
    async def test_same_spec_different_instance_same_tag(self) -> None:
        """Two independently constructed specs with same layers produce same tag."""
        from containerspec import ImageSpec

        spec_a = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("httpx")
            .entrypoint([])
        )
        spec_b = (
            ImageSpec.from_registry("python:3.12-slim", pin_digest=False)
            .pip_install("httpx")
            .entrypoint([])
        )
        assert spec_a.tag("test", client=None) == spec_b.tag("test", client=None)

        built_a = await spec_a.build("containerspec-test-cache")
        built_b = await spec_b.build("containerspec-test-cache")
        assert built_a.tag == built_b.tag

        subprocess.run(["docker", "rmi", built_a.tag], check=False, timeout=30)  # noqa: S603, S607


class TestMultiStageBuild:
    @pytest.mark.asyncio
    async def test_multistage_node_nginx(self, tmp_path: Path) -> None:
        """Build a multi-stage image: node builder compiles, nginx serves."""
        from containerspec import ImageSpec

        index_html = tmp_path / "index.html"
        index_html.write_text("<h1>Hello from containerspec</h1>")

        builder = (
            ImageSpec.from_registry("node:22-slim", pin_digest=False)
            .workdir("/app")
            .copy(str(index_html), "/app/dist/index.html")
        )
        stage = builder.with_stage("builder")

        runtime = (
            ImageSpec.from_registry("nginx:alpine", pin_digest=False)
            .copy_from_stage(stage, "/app/dist", "/usr/share/nginx/html")
            .expose(80)
            .entrypoint(["nginx", "-g", "daemon off;"])
        )
        built = await runtime.build("containerspec-test-multistage")
        assert built.tag.startswith("containerspec-test-multistage:sha-")

        result = subprocess.run(  # noqa: S603, S607
            ["docker", "run", "--rm", "--entrypoint", "", built.tag, "cat", "/usr/share/nginx/html/index.html"],
            capture_output=True, timeout=30, check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert b"Hello from containerspec" in result.stdout

        subprocess.run(["docker", "rmi", built.tag], check=False, timeout=30)  # noqa: S603, S607
