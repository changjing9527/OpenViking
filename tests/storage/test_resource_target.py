# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for the AGFS resource target used by content tree actions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclEntry, AclLevel, AclMode, DirectAcl, EffectiveAcl
from openviking.storage.resource_target import AgfsResourceTarget
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
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


@pytest.fixture
async def acl_target(tmp_path):
    grants = {}

    async def resolve_many(uris, ctx):
        return {
            uri: EffectiveAcl(
                AclMode.RESTRICTED,
                DirectAcl((AclEntry("user:alice", grants.get(uri, AclLevel.WRITE)),)),
                DirectAcl(),
            )
            for uri in uris
        }

    acl = SimpleNamespace(is_enabled=AsyncMock(return_value=True), resolve_many=resolve_many)
    agfs = MockLocalAGFS(root_path=tmp_path)
    fs = VikingFS(agfs=agfs, acl_manager=acl)
    ctx = RequestContext(user=UserIdentifier("acct", "alice"), role=Role.USER)
    root_path = fs._uri_to_path(_ROOT, ctx=ctx)
    agfs.mkdir(root_path)
    lease = await fs._async_agfs.pathlock_acquire_tree(root_path)
    target = AgfsResourceTarget(viking_fs=fs, root_uri=_ROOT, ctx=ctx, lease_ref=lease)
    yield SimpleNamespace(target=target, grants=grants, root=agfs._resolve(root_path))
    await fs._async_agfs.pathlock_release(lease)


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

    async def test_import_file_cleanup_requires_write(self, acl_target) -> None:
        old_file = acl_target.root / "gone.py"
        sibling = acl_target.root / "keep.py"
        old_file.write_bytes(b"old content")
        sibling.write_bytes(b"keep")
        acl_target.grants[f"{_ROOT}/gone.py"] = AclLevel.READ

        with pytest.raises(PermissionDeniedError, match="write permission required"):
            await acl_target.target.delete_path("gone.py", is_dir=False)
        assert old_file.read_bytes() == b"old content"

        acl_target.grants[f"{_ROOT}/gone.py"] = AclLevel.WRITE
        await acl_target.target.delete_path("gone.py", is_dir=False)
        assert not old_file.exists()
        assert sibling.read_bytes() == b"keep"

    async def test_import_directory_cleanup_requires_write_on_descendants(self, acl_target) -> None:
        nested = acl_target.root / "old-dir" / "nested"
        nested.mkdir(parents=True)
        old_file = nested / "document.md"
        old_file.write_bytes(b"old content")
        child_uri = f"{_ROOT}/old-dir/nested/document.md"
        acl_target.grants[child_uri] = AclLevel.READ

        with pytest.raises(PermissionDeniedError, match="write permission required"):
            await acl_target.target.delete_path("old-dir", is_dir=True)
        assert old_file.read_bytes() == b"old content"

        acl_target.grants[child_uri] = AclLevel.WRITE
        await acl_target.target.delete_path("old-dir", is_dir=True)
        assert not (acl_target.root / "old-dir").exists()

    async def test_read_file_returns_existing_bytes(self) -> None:
        vfs = _FakeVikingFS(existing={f"{_ROOT}/a.py": b"body"})
        target = _target(vfs)

        assert await target.read_file("a.py") == b"body"
