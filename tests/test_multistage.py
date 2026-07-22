from __future__ import annotations

from containerspec import CopyFromStage, ImageSpec, StageSpec


class TestMultiStageBuild:
    def test_with_stage_returns_stage_spec(self) -> None:
        builder = ImageSpec.from_registry("node:22", pin_digest=False).workdir("/app")
        stage = builder.with_stage("builder")
        assert isinstance(stage, StageSpec)
        assert stage.name == "builder"
        assert stage.spec.base == "node:22"

    def test_copy_from_stage_layer(self) -> None:
        builder = ImageSpec.from_registry("node:22", pin_digest=False)
        stage = builder.with_stage("builder")
        runtime = ImageSpec.from_registry("nginx:alpine", pin_digest=False).copy_from_stage(
            stage, "/app/dist", "/usr/share/nginx/html"
        )
        copy_layers = [layer for layer in runtime.layers if isinstance(layer, CopyFromStage)]
        assert len(copy_layers) == 1
        assert copy_layers[0].stage_name == "builder"
        assert copy_layers[0].src == "/app/dist"
        assert copy_layers[0].dest == "/usr/share/nginx/html"

    def test_multistage_dockerfile_has_from_as(self) -> None:
        builder = ImageSpec.from_registry("node:22", pin_digest=False).workdir("/app")
        stage = builder.with_stage("builder")
        runtime = (
            ImageSpec.from_registry("nginx:alpine", pin_digest=False)
            .copy_from_stage(stage, "/app/dist", "/usr/share/nginx/html")
            .expose(80)
        )
        df = runtime.to_dockerfile()
        assert "FROM node:22 AS builder" in df
        assert "FROM nginx:alpine" in df
        assert "COPY --from=builder /app/dist /usr/share/nginx/html" in df
        assert "EXPOSE 80" in df

    def test_stage_hash_in_payload(self) -> None:
        builder = ImageSpec.from_registry("node:22", pin_digest=False).workdir("/app")
        stage = builder.with_stage("builder")
        runtime = ImageSpec.from_registry("nginx:alpine", pin_digest=False).copy_from_stage(
            stage, "/app/dist", "/usr/share/nginx/html"
        )
        payload = runtime._canonical_payload(client=None)
        assert "stages" in payload
        assert payload["stages"][0]["name"] == "builder"
        assert len(payload["stages"][0]["hash"]) == 64

    def test_stage_change_busts_cache(self) -> None:
        builder_a = ImageSpec.from_registry("node:22", pin_digest=False).workdir("/app")
        stage_a = builder_a.with_stage("builder")
        runtime_a = ImageSpec.from_registry("nginx:alpine", pin_digest=False).copy_from_stage(
            stage_a, "/app/dist", "/usr/share/nginx/html"
        )
        builder_b = (
            ImageSpec.from_registry("node:22", pin_digest=False)
            .workdir("/app")
            .run_commands("npm run build")
        )
        stage_b = builder_b.with_stage("builder")
        runtime_b = ImageSpec.from_registry("nginx:alpine", pin_digest=False).copy_from_stage(
            stage_b, "/app/dist", "/usr/share/nginx/html"
        )
        assert runtime_a.content_hash(client=None) != runtime_b.content_hash(client=None)

    def test_multiple_stages(self) -> None:
        builder1 = ImageSpec.from_registry("node:22", pin_digest=False).with_stage("deps")
        builder2 = (
            ImageSpec.from_registry("node:22", pin_digest=False)
            .workdir("/app")
            .with_stage("builder")
        )
        runtime = (
            ImageSpec.from_registry("nginx:alpine", pin_digest=False)
            .copy_from_stage(builder1, "/app/node_modules", "/usr/share/nginx/html/node_modules")
            .copy_from_stage(builder2, "/app/dist", "/usr/share/nginx/html")
        )
        df = runtime.to_dockerfile()
        assert "FROM node:22 AS deps" in df
        assert "FROM node:22 AS builder" in df
        assert "COPY --from=deps" in df
        assert "COPY --from=builder" in df
