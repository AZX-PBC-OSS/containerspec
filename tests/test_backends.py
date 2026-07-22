from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from containerspec import (
    BuildahBackend,
    BuildError,
    BuildKitBackend,
    DockerBackend,
    DockerTarget,
    FirecrackerRootfsTarget,
    MissingToolError,
)


def _ok_proc() -> AsyncMock:
    """A subprocess mock whose communicate() succeeds with empty output."""
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (b"", b"")
    return proc


class TestBuildKitBackend:
    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_solve_and_export_calls_buildx(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={"containerspec.image_spec": "{}"},
            pull=True,
        )
        cmd = list(mock_exec.call_args.args)
        assert cmd[0] == "docker"
        assert "buildx" in cmd
        assert "build" in cmd
        assert "--output" in cmd
        # Dockerfile is passed via -f <tempfile>, never via stdin.
        assert "-f" in cmd
        assert "stdin" not in mock_exec.call_args.kwargs

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_rendered_dockerfile_wins_over_cwd_dockerfile(
        self, mock_exec: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stray Dockerfile in the build context must not hijack the build."""
        (tmp_path / "Dockerfile").write_text("FROM decoy:latest\n")
        monkeypatch.chdir(tmp_path)
        built_content: list[str] = []

        def capture(*cmd: str, **kwargs: object) -> AsyncMock:
            dockerfile_arg = cmd[cmd.index("-f") + 1]
            built_content.append(Path(dockerfile_arg).read_text())
            return _ok_proc()

        mock_exec.side_effect = capture
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM rendered:latest\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={},
            pull=False,
        )
        assert built_content == ["FROM rendered:latest\n"]

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_docker_output_type(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={},
            pull=True,
        )
        cmd = list(mock_exec.call_args.args)
        output_idx = cmd.index("--output")
        assert cmd[output_idx + 1] == "type=docker"

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_oci_output_type(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="oci",
            output_path="/tmp/tar.tar",
            labels={},
            pull=True,
        )
        cmd = list(mock_exec.call_args.args)
        output_idx = cmd.index("--output")
        assert cmd[output_idx + 1] == "type=oci,dest=/tmp/tar.tar"

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_local_output_type(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="local",
            output_path="/tmp/rootfs",
            labels={},
            pull=True,
        )
        cmd = list(mock_exec.call_args.args)
        output_idx = cmd.index("--output")
        assert cmd[output_idx + 1] == "type=local,dest=/tmp/rootfs"

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_raises_builderror_on_nonzero_exit(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate.return_value = (b"", b"buildx exploded")
        mock_exec.return_value = proc
        backend = BuildKitBackend()
        with pytest.raises(BuildError, match="buildx failed"):
            await backend.solve_and_export(
                dockerfile="FROM base\n",
                tag="x:sha-abc",
                output_type="docker",
                output_path=None,
                labels={},
                pull=True,
            )

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_dockerfile_via_temp_file_not_stdin(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={},
            pull=True,
        )
        cmd = list(mock_exec.call_args.args)
        f_idx = cmd.index("-f")
        temp_path = cmd[f_idx + 1]
        # The -f argument is a path to a now-cleaned-up temp Dockerfile.
        assert isinstance(temp_path, str)
        assert "containerspec-" in temp_path
        # No stdin piping — the dockerfile travels via -f.
        assert "stdin" not in mock_exec.call_args.kwargs

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_pull_flag_added(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={},
            pull=True,
        )
        cmd = list(mock_exec.call_args.args)
        assert "--pull" in cmd

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_no_pull_flag_when_false(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={},
            pull=False,
        )
        cmd = list(mock_exec.call_args.args)
        assert "--pull" not in cmd

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_labels_passed_as_label_flags(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend()
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={"containerspec.image_spec": "{}", "foo": "bar"},
            pull=False,
        )
        cmd = list(mock_exec.call_args.args)
        assert "--label" in cmd
        label_idx = cmd.index("--label")
        assert cmd[label_idx + 1] == "containerspec.image_spec={}"
        assert cmd[label_idx + 3] == "foo=bar"

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_builder_flag_when_set(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        backend = BuildKitBackend(builder="mybuilder")
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={},
            pull=False,
        )
        cmd = list(mock_exec.call_args.args)
        builder_idx = cmd.index("--builder")
        assert cmd[builder_idx + 1] == "mybuilder"


class TestBuildahBackend:
    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_calls_buildah_bud_then_push_for_oci(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"):
            backend = BuildahBackend()
            await backend.solve_and_export(
                dockerfile="FROM base\n",
                tag="x:sha-abc",
                output_type="oci",
                output_path="/tmp/tar.tar",
                labels={"foo": "bar"},
                pull=True,
            )
        assert mock_exec.await_count == 2
        bud_cmd = list(mock_exec.call_args_list[0].args)
        assert bud_cmd[0] == "buildah"
        assert "bud" in bud_cmd
        assert "-f" in bud_cmd
        t_idx = bud_cmd.index("-t")
        assert bud_cmd[t_idx + 1] == "x:sha-abc"
        assert "--pull" in bud_cmd
        assert "--label" in bud_cmd
        label_idx = bud_cmd.index("--label")
        assert bud_cmd[label_idx + 1] == "foo=bar"
        assert bud_cmd[-1] == "."

        push_cmd = list(mock_exec.call_args_list[1].args)
        assert push_cmd[0] == "buildah"
        assert "push" in push_cmd
        assert "oci-archive:/tmp/tar.tar" in push_cmd
        assert "x:sha-abc" in push_cmd

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_oci_push_uses_valid_buildah_argv(self, mock_exec: MagicMock) -> None:
        """buildah push must use `-f oci` (a manifest type) and an oci-archive: dest.

        `-f oci-archive` is invalid (oci-archive is a transport, not a format),
        and the destination needs the `oci-archive:` transport prefix or buildah
        pushes to a registry instead of writing a tarball.
        """
        mock_exec.return_value = _ok_proc()
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"):
            await BuildahBackend().solve_and_export(
                dockerfile="FROM base\n",
                tag="x:sha-abc",
                output_type="oci",
                output_path="/tmp/tar.tar",
                labels={},
                pull=False,
            )
        push_cmd = list(mock_exec.call_args_list[1].args)
        assert push_cmd == [
            "buildah",
            "push",
            "-f",
            "oci",
            "x:sha-abc",
            "oci-archive:/tmp/tar.tar",
        ]

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_oci_output_path_with_colon_raises(self, mock_exec: MagicMock) -> None:
        """A ':' in the OCI output path would be mis-split by the oci-archive transport.

        buildah's oci-archive:path[:reference] parses at the first colon, so a
        colon in the path silently truncates it — reject before invoking buildah.
        """
        mock_exec.return_value = _ok_proc()
        with (
            patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"),
            pytest.raises(BuildError, match="colon"),
        ):
            await BuildahBackend().solve_and_export(
                dockerfile="FROM base\n",
                tag="x:sha-abc",
                output_type="oci",
                output_path="/tmp/build:v2/out.tar",
                labels={},
                pull=False,
            )
        # The bud command may run, but the push must never fire with a bad dest.
        push_calls = [c for c in mock_exec.call_args_list if "push" in c.args]
        assert push_calls == []

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_builds_passed_dockerfile_not_context_dockerfile(
        self, mock_exec: MagicMock, tmp_path: Path
    ) -> None:
        """Buildah must build the passed dockerfile arg, never sniff the context dir.

        A Dockerfile sitting in context_path must not hijack the build — the same
        failure class that C1 fixed for BuildKit. build() forwards the rewritten
        Dockerfile as the arg; the backend must hand buildah exactly that.
        """
        (tmp_path / "Dockerfile").write_text("FROM decoy:latest\n")
        built_content: list[str] = []

        def capture(*cmd: str, **kwargs: object) -> AsyncMock:
            if "bud" in cmd:
                built_content.append(Path(cmd[cmd.index("-f") + 1]).read_text())
            return _ok_proc()

        mock_exec.side_effect = capture
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"):
            backend = BuildahBackend()
            await backend.solve_and_export(
                dockerfile="FROM rendered:latest\n",
                tag="x:sha-abc",
                output_type="oci",
                output_path="/tmp/tar.tar",
                labels={},
                pull=False,
                context_path=str(tmp_path),
            )
        assert built_content == ["FROM rendered:latest\n"]
        bud_cmd = list(mock_exec.call_args_list[0].args)
        assert bud_cmd[-1] == str(tmp_path)

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_raises_builderror_for_docker_output(self, mock_exec: MagicMock) -> None:
        mock_exec.return_value = _ok_proc()
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"):
            backend = BuildahBackend()
            with pytest.raises(BuildError, match="output_type='docker'"):
                await backend.solve_and_export(
                    dockerfile="FROM base\n",
                    tag="x:sha-abc",
                    output_type="docker",
                    output_path=None,
                    labels={},
                    pull=True,
                )
        # The docker output_type is rejected before any subprocess runs.
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_missing_tool_when_buildah_absent(self) -> None:
        with patch("containerspec.rootfs.shutil.which", return_value=None):
            backend = BuildahBackend()
            with pytest.raises(MissingToolError, match="buildah not found"):
                await backend.solve_and_export(
                    dockerfile="FROM base\n",
                    tag="x:sha-abc",
                    output_type="oci",
                    output_path="/tmp/tar.tar",
                    labels={},
                    pull=True,
                )

    @patch("containerspec.backends.asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_raises_builderror_on_nonzero_exit(self, mock_exec: MagicMock) -> None:
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate.return_value = (b"", b"buildah blew up")
        mock_exec.return_value = proc
        with patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"):
            backend = BuildahBackend()
            with pytest.raises(BuildError, match=r"buildah\.bud failed"):
                await backend.solve_and_export(
                    dockerfile="FROM base\n",
                    tag="x:sha-abc",
                    output_type="oci",
                    output_path="/tmp/tar.tar",
                    labels={},
                    pull=False,
                )

    @pytest.mark.asyncio
    async def test_local_output_type_calls_export_filesystem(self) -> None:
        with (
            patch(
                "containerspec.backends._run_command",
                new_callable=AsyncMock,
                return_value=(0, b"", b""),
            ),
            patch.object(
                BuildahBackend, "_export_filesystem", new_callable=AsyncMock
            ) as mock_export,
            patch("containerspec.rootfs.shutil.which", return_value="/usr/bin/buildah"),
        ):
            backend = BuildahBackend()
            await backend.solve_and_export(
                dockerfile="FROM base\n",
                tag="x:sha-abc",
                output_type="local",
                output_path="/tmp/rootfs",
                labels={},
                pull=False,
            )
        mock_export.assert_awaited_once_with(tag="x:sha-abc", dest="/tmp/rootfs")

    @pytest.mark.asyncio
    async def test_export_filesystem_mount_copy_umount_rm_sequence(self) -> None:
        backend = BuildahBackend()
        with patch("containerspec.backends._run_command", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = [
                (0, b"my-container\n", b""),
                (0, b"/mnt/point\n", b""),
                (0, b"", b""),
                (0, b"", b""),
                (0, b"", b""),
            ]
            await backend._export_filesystem(tag="x:sha-abc", dest="/tmp/rootfs")

        assert mock_run.await_count == 5
        cmds = [call.args[0] for call in mock_run.call_args_list]
        assert cmds[0] == ["buildah", "from", "x:sha-abc"]
        assert cmds[1] == ["buildah", "mount", "my-container"]
        assert cmds[2] == ["cp", "-a", "/mnt/point/.", "/tmp/rootfs/"]
        assert cmds[3] == ["buildah", "umount", "my-container"]
        assert cmds[4] == ["buildah", "rm", "my-container"]
        labels = [call.kwargs["label"] for call in mock_run.call_args_list]
        assert labels == [
            "buildah.from",
            "buildah.mount",
            "buildah.copy",
            "buildah.umount",
            "buildah.rm",
        ]


class TestDockerBackend:
    @pytest.mark.asyncio
    async def test_build_calls_images_build(self) -> None:
        client = MagicMock()
        client.images.build.return_value = (MagicMock(), [])
        backend = DockerBackend(client=client)
        await backend.solve_and_export(
            dockerfile="FROM base\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={"containerspec.image_spec": "{}"},
            pull=True,
        )
        client.images.build.assert_called_once()
        kwargs = client.images.build.call_args.kwargs
        assert kwargs["tag"] == "x:sha-abc"
        assert kwargs["pull"] is True
        assert kwargs["rm"] is True
        assert kwargs["labels"] == {"containerspec.image_spec": "{}"}

    @pytest.mark.asyncio
    async def test_staged_context_uses_context_dockerfile(self, tmp_path: Path) -> None:
        """With a staged context, the daemon must build the context's own Dockerfile.

        docker-py resolves ``dockerfile=`` inside the context tarball, so an
        absolute host path there is never found.
        """
        (tmp_path / "Dockerfile").write_text("FROM base\nCOPY ctx_ab12cd34_app.py /app.py\n")
        client = MagicMock()
        client.images.build.return_value = (MagicMock(), [])
        backend = DockerBackend(client=client)
        await backend.solve_and_export(
            dockerfile="FROM base\nCOPY ./app.py /app.py\n",
            tag="x:sha-abc",
            output_type="docker",
            output_path=None,
            labels={},
            pull=False,
            context_path=str(tmp_path),
        )
        kwargs = client.images.build.call_args.kwargs
        assert kwargs["path"] == str(tmp_path)
        assert kwargs["dockerfile"] == "Dockerfile"

    @pytest.mark.asyncio
    async def test_raises_builderror_for_non_docker_output(self) -> None:
        client = MagicMock()
        backend = DockerBackend(client=client)
        with pytest.raises(BuildError, match="output_type='oci'"):
            await backend.solve_and_export(
                dockerfile="FROM base\n",
                tag="x:sha-abc",
                output_type="oci",
                output_path="/tmp/tar.tar",
                labels={},
                pull=True,
            )
        client.images.build.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_client_lazily_when_none(self) -> None:
        fake_client = MagicMock()
        fake_client.images.build.return_value = (MagicMock(), [])
        fake_docker = MagicMock()
        fake_docker.from_env.return_value = fake_client
        backend = DockerBackend(client=None)
        with patch.dict(sys.modules, {"docker": fake_docker}):
            await backend.solve_and_export(
                dockerfile="FROM base\n",
                tag="x:sha-abc",
                output_type="docker",
                output_path=None,
                labels={},
                pull=True,
            )
        fake_docker.from_env.assert_called_once()
        assert backend.client is fake_client
        fake_client.images.build.assert_called_once()


class TestAutoDetect:
    @patch("containerspec.backends.shutil.which")
    def test_docker_target_uses_buildkit_when_docker_available(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/bin/docker"
        from containerspec.backends import auto_detect_backend

        backend = auto_detect_backend(target=DockerTarget(name="x"))
        assert isinstance(backend, BuildKitBackend)

    @patch("containerspec.backends.shutil.which")
    def test_docker_target_falls_back_to_dockerbackend(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        from containerspec.backends import auto_detect_backend

        backend = auto_detect_backend(target=DockerTarget(name="x"))
        assert isinstance(backend, DockerBackend)

    @patch("containerspec.backends.shutil.which")
    def test_rootfs_target_uses_buildah_when_available(self, mock_which: MagicMock) -> None:
        def which(cmd: str) -> str | None:
            return "/usr/bin/buildah" if cmd == "buildah" else None

        mock_which.side_effect = which
        from containerspec.backends import auto_detect_backend

        backend = auto_detect_backend(target=FirecrackerRootfsTarget(path="/tmp/rootfs.ext4"))
        assert isinstance(backend, BuildahBackend)

    @patch("containerspec.backends.shutil.which")
    def test_rootfs_target_falls_back_to_buildkit(self, mock_which: MagicMock) -> None:
        def which(cmd: str) -> str | None:
            return "/usr/bin/docker" if cmd == "docker" else None

        mock_which.side_effect = which
        from containerspec.backends import auto_detect_backend

        backend = auto_detect_backend(target=FirecrackerRootfsTarget(path="/tmp/rootfs.ext4"))
        assert isinstance(backend, BuildKitBackend)

    @patch("containerspec.backends.shutil.which")
    def test_rootfs_target_falls_back_to_dockerbackend(self, mock_which: MagicMock) -> None:
        mock_which.return_value = None
        from containerspec.backends import auto_detect_backend

        backend = auto_detect_backend(target=FirecrackerRootfsTarget(path="/tmp/rootfs.ext4"))
        assert isinstance(backend, DockerBackend)

    @patch("containerspec.backends.shutil.which")
    def test_no_target_defaults_to_docker_flow(self, mock_which: MagicMock) -> None:
        mock_which.return_value = "/usr/bin/docker"
        from containerspec.backends import auto_detect_backend

        backend = auto_detect_backend()
        assert isinstance(backend, BuildKitBackend)
