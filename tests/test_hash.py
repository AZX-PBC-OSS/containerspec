from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from containerspec import ImageSpec


class TestImageSpecHash:
    def test_same_spec_same_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        b = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_reordered_apt_packages_same_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).apt_install("git", "build-essential")
        b = ImageSpec.from_registry("base", pin_digest=False).apt_install("build-essential", "git")
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_reordered_layers_different_hash(self) -> None:
        a = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="w")
            .chown("/foo")
        )
        b = (
            ImageSpec.from_registry("base", pin_digest=False)
            .chown("/foo", uid=1000, gid=1000)
            .user(uid=1000, gid=1000, name="w")
        )
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_uv_vs_pip_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        b = ImageSpec.from_registry("base", pin_digest=False).pip_install("vllm")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_apt_vs_apk_different_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).apt_install("pkg")
        b = ImageSpec.from_registry("base", pin_digest=False).apk_install("pkg")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_different_base_different_hash(self) -> None:
        a = ImageSpec.from_registry("base-a", pin_digest=False)
        b = ImageSpec.from_registry("base-b", pin_digest=False)
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_pin_digest_false_no_client_needed(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False)
        h = spec.content_hash(client=None)
        assert len(h) == 64

    def test_pin_digest_false_skips_resolution(self) -> None:
        client = MagicMock()
        spec = ImageSpec.from_registry("base", pin_digest=False)
        spec.content_hash(client=client)
        client.images.get_registry_data.assert_not_called()

    def test_pin_digest_resolved_digest_feeds_hash(self) -> None:
        client = MagicMock()
        client.images.get_registry_data.side_effect = [
            MagicMock(id="sha256:aaa111"),
            MagicMock(id="sha256:bbb222"),
        ]
        a = ImageSpec.from_registry("base", pin_digest=True)
        b = ImageSpec.from_registry("base", pin_digest=True)
        assert a.content_hash(client=client) != b.content_hash(client=client)

    def test_from_registry_does_not_resolve_until_hash(self) -> None:
        client = MagicMock()
        _ = ImageSpec.from_registry("base", pin_digest=True)
        client.images.get_registry_data.assert_not_called()

    def test_chown_default_uid_null_in_payload(self) -> None:
        a = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=1000, gid=1000, name="w")
            .chown("/foo")
        )
        b = (
            ImageSpec.from_registry("base", pin_digest=False)
            .user(uid=9999, gid=9999, name="x")
            .chown("/foo")
        )
        a_payload = a._canonical_payload(client=None)
        b_payload = b._canonical_payload(client=None)
        a_chown = next(layer for layer in a_payload["layers"] if layer["type"] == "chown")
        b_chown = next(layer for layer in b_payload["layers"] if layer["type"] == "chown")
        assert a_chown["uid"] is None
        assert b_chown["uid"] is None

    def test_chown_explicit_uid_zero_in_payload(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).chown("/foo", uid=0, gid=0)
        payload = spec._canonical_payload(client=None)
        chown = next(layer for layer in payload["layers"] if layer["type"] == "chown")
        assert chown["uid"] == 0
        assert chown["gid"] == 0

    def test_user_uid_feeds_hash(self) -> None:
        a = ImageSpec.from_registry("base", pin_digest=False).user(uid=1000, gid=1000, name="w")
        b = ImageSpec.from_registry("base", pin_digest=False).user(uid=1001, gid=1000, name="w")
        assert a.content_hash(client=None) != b.content_hash(client=None)

    def test_tag_format(self) -> None:
        spec = ImageSpec.from_registry("base", pin_digest=False).uv_pip_install("vllm")
        tag = spec.tag("warden/vllm", client=None)
        assert tag.startswith("warden/vllm:sha-")
        assert len(tag.split("sha-")[1]) == 16

    def test_pin_digest_true_no_client_auto_downgrades(self) -> None:
        """pin_digest=True with no client auto-downgrades to tag-string hashing (no crash)."""
        spec = ImageSpec.from_registry("base", pin_digest=True)
        h = spec.content_hash(client=None)
        assert len(h) == 64  # works, doesn't raise

    def test_distro_parameter_sets_context(self) -> None:
        """from_registry(distro='alpine') doesn't change hash but enables correct rendering."""
        a = ImageSpec.from_registry("base", pin_digest=False, distro="alpine")
        b = ImageSpec.from_registry("base", pin_digest=False)
        assert a.content_hash(client=None) == b.content_hash(client=None)

    def test_distro_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="distro must be one of"):
            ImageSpec.from_registry("base", pin_digest=False, distro="solaris")


class TestGoldenHash:
    """Locks the cache contract. See spec for full rationale."""

    _GOLDEN_SPEC = (
        ImageSpec.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", pin_digest=False)
        .add_python("3.12")
        .apt_install("git", "build-essential")
        .uv_pip_install("vllm", "flashinfer==0.2.0")
        .env({"HF_HOME": "/home/warden/.cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .workdir("/app")
        .user(uid=1000, gid=1000, name="warden")
        .chown("/home/warden/.cache/huggingface")
        .entrypoint([])
    )

    _GOLDEN_CANONICAL_JSON = (
        '{"base": {"ref": "nvidia/cuda:13.3.0-devel-ubuntu24.04"}, '
        '"layers": ['
        '{"type": "add_python", "version": "3.12"}, '
        '{"packages": ["build-essential", "git"], "type": "apt_install"}, '
        '{"packages": ["flashinfer==0.2.0", "vllm"], "type": "uv_pip_install"}, '
        '{"type": "env", "vars": {"HF_HOME": "/home/warden/.cache/huggingface", '
        '"HF_HUB_ENABLE_HF_TRANSFER": "1"}}, '
        '{"path": "/app", "type": "workdir"}, '
        '{"alpine": false, "gid": 1000, "name": "warden", "type": "user", "uid": 1000}, '
        '{"gid": null, "path": "/home/warden/.cache/huggingface", "type": "chown", "uid": null}, '
        '{"commands": [], "type": "entrypoint"}'
        '], "pin_digest": false}'
    )

    _GOLDEN_HASH = "21b504478c60739f"

    def test_golden_canonical_payload(self) -> None:
        actual = self._GOLDEN_SPEC._canonical_json(client=None)
        assert actual == self._GOLDEN_CANONICAL_JSON, (
            f"Canonical payload drift.\nExpected: {self._GOLDEN_CANONICAL_JSON}\nActual:   {actual}"
        )

    def test_golden_hash_stable(self) -> None:
        if self._GOLDEN_HASH == "REPLACE_ME_WITH_ACTUAL_HASH":
            actual = self._GOLDEN_SPEC.content_hash(client=None)[:16]
            pytest.fail(f"Replace _GOLDEN_HASH with: {actual}")
        actual = self._GOLDEN_SPEC.content_hash(client=None)[:16]
        assert actual == self._GOLDEN_HASH, (
            f"Hash drift. Expected: {self._GOLDEN_HASH}, Actual: {actual}"
        )
