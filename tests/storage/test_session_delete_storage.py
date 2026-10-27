# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unindexed deletion avoids vector work while retaining storage safety checks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.core.namespace import resolve_request_uri
from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import AclEntry, AclLevel, AclMode, DirectAcl, EffectiveAcl
from openviking.storage.errors import LockAcquisitionError, ResourceBusyError
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import InvalidArgumentError, PermissionDeniedError
from openviking_cli.session.user_id import UserIdentifier

SESSION_URI = "viking://user/alice/sessions/session-1"
SESSION_PATH = "/local/account/user/alice/sessions/session-1"
RESOURCE_URI = "viking://resources/project"
RESOURCE_PATH = "/local/account/resources/project"


@pytest.fixture
def storage():
    """Use real URI, authorization, lock-context, and vector cleanup behavior."""
    lease = {"lease_ref": "delete-lease"}
    agfs = SimpleNamespace(
        stat=AsyncMock(return_value={"isDir": True}),
        ls=AsyncMock(return_value=[]),
        pathlock_acquire_tree=AsyncMock(return_value=lease),
        pathlock_acquire_exact=AsyncMock(return_value=lease),
        pathlock_release=AsyncMock(),
        rm=AsyncMock(return_value={}),
    )
    vector = SimpleNamespace(count=AsyncMock(return_value=7), delete_uris=AsyncMock())
    acl = SimpleNamespace(is_enabled=AsyncMock(return_value=False), resolve_many=AsyncMock())
    fs = VikingFS(agfs=SimpleNamespace(), vector_store=vector, acl_manager=acl)
    fs._async_agfs = agfs
    fs._collect_uris = AsyncMock(return_value=[])
    ctx = RequestContext(user=UserIdentifier("account", "alice"), role=Role.USER)
    return SimpleNamespace(fs=fs, agfs=agfs, vector=vector, acl=acl, ctx=ctx, lease=lease)


@pytest.mark.parametrize("acl_enabled", [False, True])
@pytest.mark.parametrize(
    "request_uri",
    [
        SESSION_URI,
        "viking://~/sessions/session-1",
        "viking://temp/alice/raw-upload-1",
    ],
)
async def test_no_vector_rm_skips_subtree_enumeration_and_vector_store(
    storage, acl_enabled, request_uri
):
    storage.acl.is_enabled.return_value = acl_enabled
    uri = resolve_request_uri(request_uri, storage.ctx)
    path = storage.fs._uri_to_path(uri, ctx=storage.ctx)

    result = await storage.fs.rm(uri, recursive=True, ctx=storage.ctx, no_vector=True)

    assert result == {"estimated_deleted_count": 0}
    storage.fs._collect_uris.assert_not_awaited()
    storage.agfs.ls.assert_not_awaited()
    storage.vector.count.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()
    storage.agfs.pathlock_acquire_tree.assert_awaited_once_with(path)
    storage.agfs.pathlock_acquire_exact.assert_not_awaited()
    storage.agfs.rm.assert_awaited_once_with(
        path,
        recursive=True,
        fs_ctx={"account_id": "account", "lease_ref": "delete-lease"},
        auto_pathlock=True,
    )
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


async def test_session_file_rm_keeps_exact_lock_and_skips_vector_store(storage):
    storage.agfs.stat.return_value = {"isDir": False}
    uri = f"{SESSION_URI}/messages.jsonl"
    path = f"{SESSION_PATH}/messages.jsonl"

    result = await storage.fs.rm(uri, ctx=storage.ctx, no_vector=True)

    assert result == {"estimated_deleted_count": 0}
    storage.fs._collect_uris.assert_not_awaited()
    storage.vector.count.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()
    storage.agfs.pathlock_acquire_tree.assert_not_awaited()
    storage.agfs.pathlock_acquire_exact.assert_awaited_once_with(path)
    storage.agfs.rm.assert_awaited_once_with(
        path,
        recursive=False,
        fs_ctx={"account_id": "account", "lease_ref": "delete-lease"},
        auto_pathlock=True,
    )
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


async def test_session_directory_rm_still_requires_recursive(storage):
    with pytest.raises(InvalidArgumentError, match="recursive"):
        await storage.fs.rm(SESSION_URI, ctx=storage.ctx, no_vector=True)

    storage.agfs.pathlock_acquire_tree.assert_not_awaited()
    storage.agfs.pathlock_acquire_exact.assert_not_awaited()
    storage.agfs.rm.assert_not_awaited()
    storage.vector.count.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()


@pytest.mark.parametrize("acl_enabled", [False, True])
@pytest.mark.parametrize("uri", [SESSION_URI, "viking://temp/alice/raw-upload-1"])
async def test_missing_no_vector_rm_does_not_scan_or_clean_vector_records(
    storage, acl_enabled, uri
):
    storage.acl.is_enabled.return_value = acl_enabled
    storage.agfs.stat.side_effect = FileNotFoundError(SESSION_PATH)

    result = await storage.fs.rm(uri, recursive=True, ctx=storage.ctx, no_vector=True)

    assert result == {"estimated_deleted_count": 0}
    storage.fs._collect_uris.assert_not_awaited()
    storage.vector.count.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()
    storage.agfs.pathlock_acquire_tree.assert_not_awaited()
    storage.agfs.rm.assert_not_awaited()


async def test_session_rm_releases_its_tree_lease_when_backend_delete_fails(storage):
    storage.agfs.rm.side_effect = OSError("storage delete failed")

    with pytest.raises(OSError, match="storage delete failed"):
        await storage.fs.rm(SESSION_URI, recursive=True, ctx=storage.ctx, no_vector=True)

    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)
    assert storage.agfs.rm.await_args.kwargs["fs_ctx"]["lease_ref"] == "delete-lease"


async def test_session_rm_rechecks_existence_after_waiting_for_tree_lock(storage):
    storage.agfs.stat.side_effect = [
        {"isDir": True},
        FileNotFoundError(SESSION_PATH),
    ]

    result = await storage.fs.rm(SESSION_URI, recursive=True, ctx=storage.ctx, no_vector=True)

    assert result == {"estimated_deleted_count": 0}
    storage.agfs.pathlock_acquire_tree.assert_awaited_once_with(SESSION_PATH)
    storage.agfs.rm.assert_not_awaited()
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


async def test_session_rm_reuses_outer_lease_without_releasing_it(storage):
    outer_lease = {"lease_ref": "outer-lease"}

    await storage.fs.rm(
        SESSION_URI, recursive=True, ctx=storage.ctx, lease_ref=outer_lease, no_vector=True
    )

    storage.agfs.pathlock_acquire_tree.assert_not_awaited()
    storage.agfs.pathlock_release.assert_not_awaited()
    assert storage.agfs.rm.await_args.kwargs["fs_ctx"]["lease_ref"] == "outer-lease"


async def test_session_rm_preserves_retryable_busy_error(storage):
    storage.agfs.pathlock_acquire_tree.side_effect = LockAcquisitionError("path is locked")

    with pytest.raises(ResourceBusyError) as error:
        await storage.fs.rm(SESSION_URI, recursive=True, ctx=storage.ctx, no_vector=True)

    assert error.value.uri == SESSION_URI
    assert error.value.conflict_type == "path_busy"
    assert error.value.retryable is True
    storage.agfs.rm.assert_not_awaited()
    storage.agfs.pathlock_release.assert_not_awaited()


@pytest.mark.parametrize("acl_enabled", [False, True])
@pytest.mark.parametrize(
    "uri",
    ["viking://user/bob/sessions/session-1", f"{SESSION_URI}/../another-session"],
)
async def test_session_rm_preserves_namespace_and_path_validation(storage, acl_enabled, uri):
    storage.acl.is_enabled.return_value = acl_enabled

    with pytest.raises(PermissionDeniedError):
        await storage.fs.rm(uri, recursive=True, ctx=storage.ctx, no_vector=True)

    storage.agfs.stat.assert_not_awaited()
    storage.agfs.pathlock_acquire_tree.assert_not_awaited()
    storage.agfs.rm.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("uri", [RESOURCE_URI, SESSION_URI])
async def test_default_rm_still_cleans_vector_records(storage, missing, uri):
    child_uri = f"{uri}/document.md"
    storage.fs._collect_uris.return_value = [child_uri]
    if missing:
        storage.agfs.stat.side_effect = FileNotFoundError(RESOURCE_PATH)

    result = await storage.fs.rm(uri, recursive=True, ctx=storage.ctx)

    assert result == {"estimated_deleted_count": 7}
    storage.fs._collect_uris.assert_awaited_once()
    storage.vector.count.assert_awaited_once()
    assert storage.vector.count.await_args.kwargs["ctx"] == storage.ctx
    storage.vector.delete_uris.assert_awaited_once_with(storage.ctx, [child_uri, uri])
    if missing:
        storage.agfs.rm.assert_not_awaited()
    else:
        storage.agfs.rm.assert_awaited_once()
        storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


@pytest.mark.parametrize("no_vector", [False, True])
@pytest.mark.parametrize("is_dir", [False, True])
async def test_resource_delete_requires_manage_on_file_and_descendants(storage, no_vector, is_dir):
    child_uri = f"{RESOURCE_URI}/restricted.md"
    uri = RESOURCE_URI if is_dir else child_uri
    storage.agfs.stat.return_value = {"isDir": is_dir}
    storage.fs._collect_uris.side_effect = lambda *args, **kwargs: [child_uri] if is_dir else []
    storage.acl.is_enabled.return_value = True
    manager = EffectiveAcl(
        AclMode.RESTRICTED,
        DirectAcl((AclEntry("user:alice", AclLevel.MANAGE),)),
        DirectAcl(),
    )
    child_acl = EffectiveAcl(
        AclMode.RESTRICTED,
        DirectAcl((AclEntry("user:alice", AclLevel.WRITE),)),
        DirectAcl(),
    )

    async def resolve_many(uris, ctx):
        return {target: child_acl if target == child_uri else manager for target in uris}

    storage.acl.resolve_many.side_effect = resolve_many
    with pytest.raises(PermissionDeniedError, match="manage permission required"):
        await storage.fs.rm(uri, recursive=is_dir, ctx=storage.ctx, no_vector=no_vector)
    storage.agfs.rm.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()

    child_acl = manager
    result = await storage.fs.rm(uri, recursive=is_dir, ctx=storage.ctx, no_vector=no_vector)

    assert result == {"estimated_deleted_count": 0 if no_vector else 7}
    storage.agfs.rm.assert_awaited_once()
    if no_vector:
        storage.vector.count.assert_not_awaited()
        storage.vector.delete_uris.assert_not_awaited()
    else:
        storage.vector.delete_uris.assert_awaited_once_with(
            storage.ctx, [child_uri, RESOURCE_URI] if is_dir else [child_uri]
        )
