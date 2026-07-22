from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from containerspec import ImageSpec
from containerspec.layers import Layer, layer_payload


def _layer_payload(spec: ImageSpec, layer_type: str) -> dict[str, Any]:
    payload = spec._canonical_payload(client=None)
    return next(layer for layer in payload["layers"] if layer["type"] == layer_type)


class TestLayerPayload:
    def test_run_commands_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).run_commands("echo hi")
        assert _layer_payload(spec, "run_commands") == {
            "type": "run_commands",
            "commands": ["echo hi"],
        }

    def test_expose_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).expose(8000)
        assert _layer_payload(spec, "expose") == {"type": "expose", "ports": [8000]}

    def test_expose_sorts_ports(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).expose(8080, 8000)
        assert _layer_payload(spec, "expose") == {"type": "expose", "ports": [8000, 8080]}

    def test_cmd_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).cmd(["vllm"])
        assert _layer_payload(spec, "cmd") == {"type": "cmd", "commands": ["vllm"]}

    def test_cmd_none_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).cmd(None)
        assert _layer_payload(spec, "cmd") == {"type": "cmd", "commands": None}

    def test_volume_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).volume("/data")
        assert _layer_payload(spec, "volume") == {"type": "volume", "paths": ["/data"]}

    def test_copy_payload(self, tmp_path) -> None:
        src = tmp_path / "file.py"
        src.write_text("print('hello')")
        spec = ImageSpec.from_registry("base", pin_digest=False).copy(str(src), "/dest.py")
        payload = layer_payload(spec.layers[0])
        assert payload["type"] == "copy"
        assert payload["src"] == str(src)
        assert payload["dest"] == "/dest.py"
        assert payload["content_hash"] != ""


class TestLayerAffectsHash:
    def test_run_commands_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.run_commands("echo hi").content_hash(
            client=None
        )

    def test_expose_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.expose(8000).content_hash(client=None)

    def test_cmd_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.cmd(["vllm"]).content_hash(client=None)

    def test_volume_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.volume("/data").content_hash(client=None)

    def test_copy_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.copy("./src", "/dest").content_hash(
            client=None
        )


class TestLayerPayloadUnknown:
    def test_unknown_layer_type_raises_typeerror(self) -> None:
        @dataclass(frozen=True)
        class Unknown(Layer):
            pass

        with pytest.raises(TypeError, match="Unknown layer type"):
            layer_payload(Unknown())


class TestNewLayerPayload:
    def test_brew_install_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).brew_install("jq", "yq")
        assert _layer_payload(spec, "brew_install") == {
            "type": "brew_install",
            "packages": ["jq", "yq"],
        }

    def test_nvm_install_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).nvm_install("20")
        assert _layer_payload(spec, "nvm_install") == {
            "type": "nvm_install",
            "version": "20",
        }

    def test_npm_install_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).npm_install(
            "typescript", "prettier"
        )
        assert _layer_payload(spec, "npm_install") == {
            "type": "npm_install",
            "packages": ["prettier", "typescript"],
        }

    def test_dnf_install_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).dnf_install("git", "curl")
        assert _layer_payload(spec, "dnf_install") == {
            "type": "dnf_install",
            "packages": ["curl", "git"],
        }


class TestNewLayerAffectsHash:
    def test_brew_install_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.brew_install("jq").content_hash(client=None)

    def test_nvm_install_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.nvm_install("20").content_hash(client=None)

    def test_npm_install_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.npm_install("typescript").content_hash(
            client=None
        )

    def test_dnf_install_affects_hash(self) -> None:
        base = ImageSpec.from_registry("base", pin_digest=False)
        assert base.content_hash(client=None) != base.dnf_install("git").content_hash(client=None)

    def test_brew_different_packages_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).brew_install("jq")
        b = ImageSpec.from_registry("base", pin_digest=False).brew_install("yq")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_npm_reordered_same_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).npm_install("typescript", "prettier")
        b = ImageSpec.from_registry("base", pin_digest=False).npm_install("prettier", "typescript")
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_dnf_reordered_same_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).dnf_install("git", "curl")
        b = ImageSpec.from_registry("base", pin_digest=False).dnf_install("curl", "git")
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_brew_vs_apt_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).brew_install("jq")
        b = ImageSpec.from_registry("base", pin_digest=False).apt_install("jq")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_dnf_vs_apt_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).dnf_install("jq")
        b = ImageSpec.from_registry("base", pin_digest=False).apt_install("jq")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_npm_vs_pip_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).npm_install("jq")
        b = ImageSpec.from_registry("base", pin_digest=False).pip_install("jq")
        assert a.content_hash(client=None) != b.content_hash(client=None)
