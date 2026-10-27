# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for the AGFS resource target used by content tree actions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.storage.acl import AclEntry, AclLevel, AclMode, DirectAcl, EffectiveAcl
from openviking.storage.resource_target import AgfsResourceTarget
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import PermissionDeniedError
from tests.utils.mock_agfs import MockLocalAGFS


class _Ctx:
    account_id = "acct"


class _FakeVikingFS:
    def __init__(self, existing=None):
        self.files = dict(existing or {})
        self.written = []

    async def write_file_bytes(self, uri, content, *, ctx=None, lease_ref=None):
        self.files[uri] = content
        self.written.append(uri)

    async def read_file_bytes(self, uri, *, ctx=None):
        return self.files[uri]


_ROOT = "viking://resources/proj"


def _target(vfs):
    return AgfsResourceTarget(viking_fs=vfs, root_uri=_ROOT, ctx=_Ctx())


@pytest.mark.asyncio
class TestAgfsResourceTarget:
    async def test_write_joins_rel_path_under_root(self) -> None:
        vfs = _FakeVikingFS()
        target = _target(vfs)

        await target.write_file("sub/a.py", b"print(1)")

        assert vfs.files[f"{_ROOT}/sub/a.py"] == b"print(1)"

    async def test_empty_rel_path_targets_root_file(self) -> None:
        vfs = _FakeVikingFS()
        target = _target(vfs)

        await target.write_file("", b"body")

        assert vfs.files[_ROOT] == b"body"

    async def test_write_preserves_artifact_bytes(self) -> None:
        vfs = _FakeVikingFS()
        target = _target(vfs)

        artifact_bytes = "你好".encode("gbk")
        await target.write_file("a.py", artifact_bytes)

        assert vfs.files[f"{_ROOT}/a.py"] == artifact_bytes

    async def test_write_rejects_path_escape(self) -> None:
        vfs = _FakeVikingFS()
        target = _target(vfs)

        with pytest.raises(ValueError):
            await target.write_file("../evil.py", b"x")

    async def test_import_directory_cleanup_requires_write_on_descendants(
        self, tmp_path, request_context
    ) -> None:
        root_path = f"/local/{request_context.account_id}/resources/proj"
        child_uri = f"{_ROOT}/nested/document.md"
        writer = EffectiveAcl(
            AclMode.RESTRICTED,
            DirectAcl((AclEntry("user:*", AclLevel.WRITE),)),
            DirectAcl(),
        )
        denied = EffectiveAcl(AclMode.RESTRICTED, DirectAcl(), DirectAcl())
        grants = {_ROOT: writer, f"{_ROOT}/nested": writer, child_uri: denied}
        acl = SimpleNamespace(
            is_enabled=AsyncMock(return_value=True), resolve_many=AsyncMock(return_value=grants)
        )
        agfs = MockLocalAGFS(root_path=tmp_path)
        agfs.write(f"{root_path}/nested/document.md", b"content")
        fs = VikingFS(agfs=agfs, acl_manager=acl)
        target = AgfsResourceTarget(viking_fs=fs, root_uri=_ROOT, ctx=request_context)

        with pytest.raises(PermissionDeniedError, match="write permission required"):
            await target.delete_path("", is_dir=True)
        assert agfs.read_file(f"{root_path}/nested/document.md") == b"content"

        grants[child_uri] = writer
        await target.delete_path("", is_dir=True)
        assert not agfs.exists(root_path)

    async def test_read_file_returns_existing_bytes(self) -> None:
        vfs = _FakeVikingFS(existing={f"{_ROOT}/a.py": b"body"})
        target = _target(vfs)

        assert await target.read_file("a.py") == b"body"
