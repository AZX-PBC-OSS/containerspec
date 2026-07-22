from __future__ import annotations

from containerspec import ImageSpec


class TestPublicApi:
    def test_to_dockerfile_no_docker(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        df = spec.to_dockerfile()
        assert isinstance(df, str)
        assert "FROM base" in df

    def test_content_hash_no_docker(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        h = spec.content_hash(client=None)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_tag_no_docker(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        tag = spec.tag("warden/vllm", client=None)
        assert tag.startswith("warden/vllm:sha-")
        assert len(tag.split("sha-")[1]) == 16

    def test_repr_exact_format(self) -> None:
        spec = ImageSpec.from_registry("base").add_python("3.12").uv_pip_install("vllm")
        assert repr(spec) == "ImageSpec(base='base', layers=2)"

    def test_all_exports_available(self) -> None:
        import containerspec

        for name in [
            "ImageSpec",
            "Layer",
            "AddPython",
            "AptInstall",
            "ApkInstall",
            "UvPipInstall",
            "PipInstall",
            "Env",
            "RunCommands",
            "Workdir",
            "Chown",
            "User",
            "Entrypoint",
            "Expose",
            "Cmd",
            "Volume",
            "Copy",
            "BuiltImage",
            "FirecrackerRootfs",
            "OciArtifact",
            "BuildTarget",
            "DockerTarget",
            "FirecrackerRootfsTarget",
            "OciTarget",
            "BuildBackend",
            "BuildKitBackend",
            "BuildahBackend",
            "DockerBackend",
            "BuildError",
            "MissingToolError",
        ]:
            assert hasattr(containerspec, name), f"Missing export: {name}"
