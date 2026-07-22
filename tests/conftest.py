"""Shared test fixtures for containerspec."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """A MagicMock that simulates docker-py's DockerClient."""
    client = MagicMock()
    return client
