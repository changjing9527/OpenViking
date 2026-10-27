# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for the AGFS resource target used by content tree actions."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclEntry, AclLevel, AclMode, DirectAcl, EffectiveAcl
from openviking.storage.context_update_plan import (
    ContentTreeAction,
    ContentTreeOperation,
    execute_content_tree_actions,
)
from openviking.storage.resource_target import AgfsResourceTarget
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier
from tests.utils.mock_agfs import MockLocalAGFS


def _ctx():
    return RequestContext(
        user=UserIdentifier("acct", "alice"),
        role=Role(Role.USER),
        group_ids=("editors",),
        actor_peer_id="peer-1",
        api_key="test-key",
        from_oauth=True,
    )


class _FakeVikingFS:
    def __init__(self, existing=None):
        self.files = dict(existing or {})
        self.written = []
        self.removed = []

    async def write_file_bytes(self, uri, content, *, ctx=None, lease_ref=None):
        self.files[uri] = content
        self.written.append(uri)

    async def read_file_bytes(self, uri, *, ctx=None):
        return self.files[uri]

    async def rm(self, uri, *, recursive=False, ctx=None, lease_ref=None):
        self.removed.append(uri)
        self.files.pop(uri, None)

    async def remove_files(self, uri, *, recursive=False, ctx=None, lease_ref=None):
        self.removed.append((uri, recursive, lease_ref))
        self.files.pop(uri, None)

    async def _ensure_access(self, uri, ctx, *, action):
        pass

    async def _acl_enabled(self, ctx):
        return False


_ROOT = "viking://resources/proj"


def _target(vfs):
    return AgfsResourceTarget(viking_fs=vfs, root_uri=_ROOT, ctx=_ctx())


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

    async def test_delete_path_uses_exact_semantics_for_file(self) -> None:
        lease = {"lease_ref": "root-tree"}
        vfs = _FakeVikingFS(existing={f"{_ROOT}/gone.py": b"x"})
        target = AgfsResourceTarget(
            viking_fs=vfs,
            root_uri=_ROOT,
            ctx=_ctx(),
            lease_ref=lease,
        )

        await target.delete_path("gone.py", is_dir=False)

        assert vfs.removed == [(f"{_ROOT}/gone.py", False, lease)]

    async def test_delete_path_uses_tree_semantics_for_directory(self) -> None:
        lease = {"lease_ref": "root-tree"}
        vfs = _FakeVikingFS()
        target = AgfsResourceTarget(
            viking_fs=vfs,
            root_uri=_ROOT,
            ctx=_ctx(),
            lease_ref=lease,
        )

        await target.delete_path("old-dir", is_dir=True)

        assert vfs.removed == [(f"{_ROOT}/old-dir", True, lease)]

    async def test_read_file_returns_existing_bytes(self) -> None:
        vfs = _FakeVikingFS(existing={f"{_ROOT}/a.py": b"body"})
        target = _target(vfs)

        assert await target.read_file("a.py") == b"body"


@pytest.fixture
async def acl_target(tmp_path, monkeypatch):
    """Exercise the real permission evaluator, tree traversal and storage delete."""
    grants: dict[str, AclLevel] = {_ROOT: AclLevel.WRITE}

    async def resolve_many(uris, ctx):
        result = {}
        for uri in uris:
            ancestor = uri
            while ancestor not in grants and ancestor.startswith("viking://resources/"):
                ancestor = ancestor.rsplit("/", 1)[0]
            entries = (AclEntry("user:alice", grants[ancestor]),) if ancestor in grants else ()
            result[uri] = EffectiveAcl(AclMode.RESTRICTED, DirectAcl(entries), DirectAcl())
        return result

    acl: Any = SimpleNamespace(is_enabled=AsyncMock(return_value=True), resolve_many=resolve_many)
    agfs = MockLocalAGFS(root_path=tmp_path)
    fs = VikingFS(agfs=agfs, acl_manager=acl)
    ctx = _ctx()
    root_path = fs._uri_to_path(_ROOT, ctx=ctx)
    agfs.mkdir(root_path)
    lease = await fs._async_agfs.pathlock_acquire_tree(root_path)
    remove = AsyncMock(wraps=fs.remove_files)
    monkeypatch.setattr(fs, "remove_files", remove)
    target = AgfsResourceTarget(viking_fs=fs, root_uri=_ROOT, ctx=ctx, lease_ref=lease)
    yield SimpleNamespace(
        fs=fs,
        agfs=agfs,
        ctx=ctx,
        lease=lease,
        grants=grants,
        target=target,
        root=agfs._resolve(root_path),
        remove=remove,
    )
    await fs._async_agfs.pathlock_release(lease)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", [ContentTreeOperation.DELETE, ContentTreeOperation.REPLACE_KIND]
)
async def test_import_cleanup_with_write_permission(acl_target, operation):
    storage = acl_target
    old_file = storage.root / "old.txt"
    old_file.write_bytes(b"old content")
    action = ContentTreeAction(
        operation,
        "old.txt",
        old_kind="file",
        new_kind="directory" if operation is ContentTreeOperation.REPLACE_KIND else None,
    )

    await execute_content_tree_actions(
        (action,), store=None, artifact_ref=None, target=storage.target
    )

    if operation is ContentTreeOperation.REPLACE_KIND:
        assert old_file.is_dir()
    else:
        assert not old_file.exists()
    cleanup_ctx = storage.remove.await_args.kwargs["ctx"]
    assert cleanup_ctx is not storage.ctx
    assert cleanup_ctx.bypass_acl
    assert not storage.ctx.bypass_acl
    assert cleanup_ctx.user is storage.ctx.user
    assert cleanup_ctx.role == storage.ctx.role
    assert cleanup_ctx.group_ids == storage.ctx.group_ids
    assert cleanup_ctx.actor_peer_id == storage.ctx.actor_peer_id
    assert cleanup_ctx.api_key == storage.ctx.api_key
    assert cleanup_ctx.from_oauth == storage.ctx.from_oauth
    assert storage.remove.await_args.kwargs["lease_ref"] is storage.lease
    assert storage.agfs._pathlock_leases


@pytest.mark.asyncio
async def test_import_cleanup_requires_write_on_exact_file(acl_target):
    storage = acl_target
    old_file = storage.root / "old.txt"
    old_file.write_bytes(b"old content")
    storage.grants[f"{_ROOT}/old.txt"] = AclLevel.READ

    with pytest.raises(PermissionDeniedError, match="write permission required"):
        await storage.target.delete_path("old.txt", is_dir=False)

    assert old_file.read_bytes() == b"old content"
    storage.remove.assert_not_awaited()
    assert not storage.ctx.bypass_acl


@pytest.mark.asyncio
@pytest.mark.parametrize("child_level", [AclLevel.WRITE, AclLevel.READ])
async def test_import_directory_cleanup_checks_complete_subtree(acl_target, child_level):
    storage = acl_target
    nested = storage.root / "old" / "nested"
    nested.mkdir(parents=True)
    (nested / "document.txt").write_bytes(b"private content")
    (storage.root / "old" / "sibling.txt").write_bytes(b"sibling content")
    storage.grants[f"{_ROOT}/old/nested/document.txt"] = child_level

    if child_level is AclLevel.READ:
        with pytest.raises(PermissionDeniedError, match="write permission required"):
            await storage.target.delete_path("old", is_dir=True)
        assert (nested / "document.txt").read_bytes() == b"private content"
        assert (storage.root / "old" / "sibling.txt").read_bytes() == b"sibling content"
        storage.remove.assert_not_awaited()
    else:
        await storage.target.delete_path("old", is_dir=True)
        assert not (storage.root / "old").exists()


@pytest.mark.asyncio
async def test_import_directory_cleanup_fails_closed_when_scan_fails(acl_target, monkeypatch):
    storage = acl_target
    (storage.root / "old").mkdir()
    old_file = storage.root / "old" / "document.txt"
    old_file.write_bytes(b"old content")
    monkeypatch.setattr(storage.fs, "_ls_entries", AsyncMock(side_effect=FileNotFoundError("scan")))

    with pytest.raises(FileNotFoundError, match="scan"):
        await storage.target.delete_path("old", is_dir=True)

    assert old_file.read_bytes() == b"old content"
    storage.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_cleanup_without_acl_does_not_require_subtree_scan(acl_target, monkeypatch):
    storage = acl_target
    (storage.root / "old").mkdir()
    (storage.root / "old" / "document.txt").write_bytes(b"old content")
    storage.fs.acl_manager.is_enabled.return_value = False
    monkeypatch.setattr(storage.fs, "_ls_entries", AsyncMock(side_effect=AssertionError("scan")))

    await storage.target.delete_path("old", is_dir=True)

    assert not (storage.root / "old").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("root_uri", "rel_path", "error"),
    [
        (_ROOT, "../escape.txt", ValueError),
        ("viking://user/bob/memories", "private.txt", PermissionDeniedError),
    ],
)
async def test_import_cleanup_preserves_path_and_namespace_checks(
    acl_target, root_uri, rel_path, error
):
    storage = acl_target
    target = AgfsResourceTarget(
        viking_fs=storage.fs, root_uri=root_uri, ctx=storage.ctx, lease_ref=storage.lease
    )

    with pytest.raises(error):
        await target.delete_path(rel_path, is_dir=False)

    storage.remove.assert_not_awaited()
