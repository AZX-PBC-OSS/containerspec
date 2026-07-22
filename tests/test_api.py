from __future__ import annotations

import pytest

from containerspec import (
    AddPython,
    ApkInstall,
    AptInstall,
    BrewInstall,
    Chown,
    Cmd,
    Copy,
    CopyFromStage,
    DnfInstall,
    Entrypoint,
    Env,
    Expose,
    ImageSpec,
    NpmInstall,
    NvmInstall,
    PipInstall,
    RunCommands,
    User,
    UvPipInstall,
    Volume,
    Workdir,
)


class TestImageSpecFluentApi:
    def test_from_registry_creates_empty_spec(self) -> None:
        spec = ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04")
        assert spec.base == "nvidia/cuda:13.3.0-devel-ubuntu24.04"
        assert spec.layers == ()

    def test_pin_digest_default_true(self) -> None:
        assert ImageSpec.from_registry("base").pin_digest is True

    def test_pin_digest_false(self) -> None:
        assert ImageSpec.from_registry("base", pin_digest=False).pin_digest is False

    def test_add_python(self) -> None:
        spec = ImageSpec.from_registry("base").add_python("3.12")
        assert isinstance(spec.layers[0], AddPython)
        assert spec.layers[0].version == "3.12"

    def test_apt_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").apt_install("git", "build-essential")
        assert isinstance(spec.layers[0], AptInstall)
        assert spec.layers[0].packages == ("build-essential", "git")

    def test_apk_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("alpine:3.20").apk_install("openrc", "util-linux")
        assert isinstance(spec.layers[0], ApkInstall)
        assert spec.layers[0].packages == ("openrc", "util-linux")

    def test_uv_pip_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").uv_pip_install("vllm", "flashinfer==0.2.0")
        assert isinstance(spec.layers[0], UvPipInstall)
        assert spec.layers[0].packages == ("flashinfer==0.2.0", "vllm")

    def test_pip_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").pip_install("httpx", "rich")
        assert isinstance(spec.layers[0], PipInstall)
        assert spec.layers[0].packages == ("httpx", "rich")

    def test_env_sorted_by_key(self) -> None:
        spec = ImageSpec.from_registry("base").env({"B": "2", "A": "1"})
        assert isinstance(spec.layers[0], Env)
        assert dict(spec.layers[0].vars) == {"A": "1", "B": "2"}

    def test_run_commands_preserves_order(self) -> None:
        spec = ImageSpec.from_registry("base").run_commands("echo a", "echo b")
        assert isinstance(spec.layers[0], RunCommands)
        assert spec.layers[0].commands == ("echo a", "echo b")

    def test_workdir(self) -> None:
        spec = ImageSpec.from_registry("base").workdir("/app")
        assert isinstance(spec.layers[0], Workdir)
        assert spec.layers[0].path == "/app"

    def test_chown_default(self) -> None:
        spec = ImageSpec.from_registry("base").chown("/foo")
        assert isinstance(spec.layers[0], Chown)
        assert spec.layers[0].uid is None
        assert spec.layers[0].gid is None

    def test_chown_explicit(self) -> None:
        spec = ImageSpec.from_registry("base").chown("/foo", uid=0, gid=0)
        assert spec.layers[0].uid == 0
        assert spec.layers[0].gid == 0

    def test_user(self) -> None:
        spec = ImageSpec.from_registry("base").user(uid=1000, gid=1000, name="warden")
        assert isinstance(spec.layers[0], User)
        assert spec.layers[0].uid == 1000

    def test_entrypoint_empty(self) -> None:
        spec = ImageSpec.from_registry("base").entrypoint([])
        assert isinstance(spec.layers[0], Entrypoint)
        assert spec.layers[0].commands == ()

    def test_entrypoint_none(self) -> None:
        spec = ImageSpec.from_registry("base").entrypoint(None)
        assert spec.layers[0].commands is None

    def test_immutability(self) -> None:
        spec = ImageSpec.from_registry("base")
        new = spec.add_python("3.12")
        assert spec.layers == ()
        assert len(new.layers) == 1

    def test_chaining(self) -> None:
        spec = (
            ImageSpec.from_registry("base")
            .add_python("3.12")
            .uv_pip_install("vllm")
            .workdir("/app")
            .user(uid=1000, gid=1000, name="warden")
            .entrypoint([])
        )
        assert len(spec.layers) == 5
        assert [type(layer).__name__ for layer in spec.layers] == [
            "AddPython",
            "UvPipInstall",
            "Workdir",
            "User",
            "Entrypoint",
        ]

    def test_repr(self) -> None:
        spec = ImageSpec.from_registry("base").add_python("3.12")
        assert repr(spec) == "ImageSpec(base='base', layers=1)"

    def test_expose_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").expose(8080, 8000, 22)
        assert isinstance(spec.layers[0], Expose)
        assert spec.layers[0].ports == (22, 8000, 8080)

    def test_cmd(self) -> None:
        spec = ImageSpec.from_registry("base").cmd(["vllm", "serve", "model"])
        assert isinstance(spec.layers[0], Cmd)
        assert spec.layers[0].commands == ("vllm", "serve", "model")

    def test_cmd_none(self) -> None:
        spec = ImageSpec.from_registry("base").cmd(None)
        assert spec.layers[0].commands is None

    def test_volume_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").volume("/data", "/cache")
        assert isinstance(spec.layers[0], Volume)
        assert spec.layers[0].paths == ("/cache", "/data")

    def test_copy(self, tmp_path) -> None:
        src = tmp_path / "init.sh"
        src.write_text("#!/bin/bash\necho hello")
        spec = ImageSpec.from_registry("base").copy(str(src), "/app/init.sh")
        assert isinstance(spec.layers[0], Copy)
        assert spec.layers[0].src == str(src)
        assert spec.layers[0].dest == "/app/init.sh"
        assert spec.layers[0].content_hash != ""

    def test_copy_content_hash_changes_with_file(self, tmp_path) -> None:
        src = tmp_path / "app.py"
        src.write_text("print('v1')")
        spec1 = ImageSpec.from_registry("base", pin_digest=False).copy(str(src), "/app/app.py")
        src.write_text("print('v2')")
        spec2 = ImageSpec.from_registry("base", pin_digest=False).copy(str(src), "/app/app.py")
        assert spec1.layers[0].content_hash != spec2.layers[0].content_hash
        assert spec1.content_hash(client=None) != spec2.content_hash(client=None)

    def test_copy_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ImageSpec.from_registry("base").copy("/nonexistent/file.py", "/app/file.py")

    def test_copy_from_stage(self) -> None:
        builder = ImageSpec.from_registry("node:22", pin_digest=False).workdir("/app")
        stage = builder.with_stage("builder")
        runtime = (
            ImageSpec.from_registry("nginx:alpine", pin_digest=False)
            .copy_from_stage(stage, "/app/dist", "/usr/share/nginx/html")
            .expose(80)
        )
        assert any(isinstance(layer, CopyFromStage) for layer in runtime.layers)
        assert runtime.layers[0].stage_name == "builder"

    def test_brew_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").brew_install("jq", "yq")
        assert isinstance(spec.layers[0], BrewInstall)
        assert spec.layers[0].packages == ("jq", "yq")

    def test_nvm_install(self) -> None:
        spec = ImageSpec.from_registry("base").nvm_install("20")
        assert isinstance(spec.layers[0], NvmInstall)
        assert spec.layers[0].version == "20"

    def test_npm_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").npm_install("typescript", "prettier")
        assert isinstance(spec.layers[0], NpmInstall)
        assert spec.layers[0].packages == ("prettier", "typescript")

    def test_dnf_install_sorted(self) -> None:
        spec = ImageSpec.from_registry("base").dnf_install("git", "curl")
        assert isinstance(spec.layers[0], DnfInstall)
        assert spec.layers[0].packages == ("curl", "git")


class TestInputValidation:
    def test_apt_install_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one package"):
            ImageSpec.from_registry("base").apt_install()

    def test_apk_install_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one package"):
            ImageSpec.from_registry("base").apk_install()

    def test_pip_install_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one package"):
            ImageSpec.from_registry("base").pip_install()

    def test_from_registry_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty base"):
            ImageSpec.from_registry("")
