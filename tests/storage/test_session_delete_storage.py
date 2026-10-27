# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Unindexed deletion avoids vector work while retaining storage safety checks."""

from dataclasses import replace
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
RESOURCE_FILE_URI = f"{RESOURCE_URI}/document.md"
RESOURCE_FILE_PATH = f"{RESOURCE_PATH}/document.md"


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


@pytest.mark.parametrize(
    "delete_method,no_vector", [("rm", False), ("rm", True), ("remove_files", True)]
)
@pytest.mark.parametrize("child_allowed", [False, True])
async def test_resource_delete_still_checks_descendant_manage_acl(
    storage, delete_method, no_vector, child_allowed
):
    child_uri = f"{RESOURCE_URI}/restricted.md"
    storage.fs._collect_uris.return_value = [child_uri]
    storage.acl.is_enabled.return_value = True
    allowed = EffectiveAcl(
        AclMode.RESTRICTED,
        DirectAcl((AclEntry("user:alice", AclLevel.MANAGE),)),
        DirectAcl(),
    )
    denied = EffectiveAcl(AclMode.RESTRICTED, DirectAcl(), DirectAcl())

    async def resolve_many(uris, ctx):
        return {uri: denied if uri == child_uri and not child_allowed else allowed for uri in uris}

    storage.acl.resolve_many.side_effect = resolve_many
    kwargs = {"no_vector": no_vector} if delete_method == "rm" else {}

    if child_allowed:
        result = await getattr(storage.fs, delete_method)(
            RESOURCE_URI, recursive=True, ctx=storage.ctx, **kwargs
        )
        assert result == {"estimated_deleted_count": 0 if no_vector else 7}
        storage.agfs.rm.assert_awaited_once()
    else:
        with pytest.raises(PermissionDeniedError, match="restricted.md"):
            await getattr(storage.fs, delete_method)(
                RESOURCE_URI, recursive=True, ctx=storage.ctx, **kwargs
            )
        storage.agfs.rm.assert_not_awaited()

    storage.fs._collect_uris.assert_awaited_once_with(
        RESOURCE_PATH, True, ctx=storage.ctx, strict=True
    )
    if no_vector or not child_allowed:
        storage.vector.count.assert_not_awaited()
        storage.vector.delete_uris.assert_not_awaited()
    else:
        storage.vector.count.assert_awaited_once()
        storage.vector.delete_uris.assert_awaited_once_with(storage.ctx, [child_uri, RESOURCE_URI])
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


@pytest.fixture
def file_storage(storage):
    storage.agfs.stat.return_value = {"isDir": False}
    storage.acl.is_enabled.return_value = True
    storage.acl_level = AclLevel.WRITE

    async def resolve_many(uris, ctx):
        effective = EffectiveAcl(
            AclMode.RESTRICTED,
            DirectAcl((AclEntry("user:alice", storage.acl_level),)),
            DirectAcl(),
        )
        return dict.fromkeys(uris, effective)

    storage.acl.resolve_many.side_effect = resolve_many
    return storage


@pytest.mark.parametrize(
    "delete_method,no_vector", [("rm", False), ("rm", True), ("remove_files", True)]
)
@pytest.mark.parametrize("level", [AclLevel.WRITE, AclLevel.MANAGE])
async def test_resource_file_delete_requires_manage(file_storage, delete_method, no_vector, level):
    storage = file_storage
    storage.acl_level = level
    kwargs = {"no_vector": no_vector} if delete_method == "rm" else {}

    if level is AclLevel.WRITE:
        with pytest.raises(PermissionDeniedError, match="manage permission required"):
            await getattr(storage.fs, delete_method)(RESOURCE_FILE_URI, ctx=storage.ctx, **kwargs)
        storage.agfs.rm.assert_not_awaited()
        storage.vector.count.assert_not_awaited()
        storage.vector.delete_uris.assert_not_awaited()
        storage.agfs.pathlock_acquire_exact.assert_not_awaited()
        return

    result = await getattr(storage.fs, delete_method)(RESOURCE_FILE_URI, ctx=storage.ctx, **kwargs)

    assert result == {"estimated_deleted_count": 0 if no_vector else 7}
    storage.agfs.rm.assert_awaited_once()
    storage.agfs.pathlock_acquire_exact.assert_awaited_once_with(RESOURCE_FILE_PATH)
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)
    if no_vector:
        storage.vector.count.assert_not_awaited()
        storage.vector.delete_uris.assert_not_awaited()
    else:
        storage.vector.delete_uris.assert_awaited_once_with(storage.ctx, [RESOURCE_FILE_URI])


@pytest.mark.parametrize("no_vector", [False, True])
@pytest.mark.parametrize("level", [AclLevel.WRITE, AclLevel.MANAGE])
async def test_missing_resource_file_delete_requires_manage(file_storage, no_vector, level):
    storage = file_storage
    storage.acl_level = level
    storage.agfs.stat.side_effect = FileNotFoundError(RESOURCE_FILE_PATH)

    if level is AclLevel.WRITE:
        with pytest.raises(PermissionDeniedError, match="manage permission required"):
            await storage.fs.rm(RESOURCE_FILE_URI, ctx=storage.ctx, no_vector=no_vector)
    else:
        result = await storage.fs.rm(RESOURCE_FILE_URI, ctx=storage.ctx, no_vector=no_vector)
        assert result == {"estimated_deleted_count": 0 if no_vector else 7}

    storage.agfs.rm.assert_not_awaited()
    storage.agfs.pathlock_acquire_exact.assert_not_awaited()
    if no_vector or level is AclLevel.WRITE:
        storage.vector.count.assert_not_awaited()
        storage.vector.delete_uris.assert_not_awaited()
    else:
        storage.vector.delete_uris.assert_awaited_once_with(storage.ctx, [RESOURCE_FILE_URI])


@pytest.mark.parametrize(
    "uri,acl_enabled",
    [
        (RESOURCE_FILE_URI, False),
        ("viking://user/alice/resources/document.md", True),
        ("viking://temp/alice/document.md", True),
    ],
)
async def test_file_delete_preserves_disabled_acl_and_private_scope_behavior(
    file_storage, uri, acl_enabled
):
    storage = file_storage
    storage.acl.is_enabled.return_value = acl_enabled

    result = await storage.fs.rm(uri, ctx=storage.ctx)

    assert result == {"estimated_deleted_count": 7}
    storage.agfs.rm.assert_awaited_once()
    storage.vector.delete_uris.assert_awaited_once_with(storage.ctx, [uri])
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


@pytest.mark.parametrize("access", ["admin", "bypass"])
async def test_resource_file_delete_preserves_admin_and_internal_acl_bypass(file_storage, access):
    storage = file_storage
    ctx = (
        replace(storage.ctx, role=Role.ADMIN)
        if access == "admin"
        else replace(storage.ctx, bypass_acl=True)
    )

    await storage.fs.rm(RESOURCE_FILE_URI, ctx=ctx)

    storage.agfs.rm.assert_awaited_once()
    storage.vector.delete_uris.assert_awaited_once_with(ctx, [RESOURCE_FILE_URI])
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


@pytest.mark.parametrize("no_vector", [False, True])
async def test_resource_file_delete_rechecks_manage_after_exact_lock(file_storage, no_vector):
    storage = file_storage
    storage.acl_level = AclLevel.MANAGE

    async def acquire_after_acl_downgrade(path):
        storage.acl_level = AclLevel.WRITE
        return storage.lease

    storage.agfs.pathlock_acquire_exact.side_effect = acquire_after_acl_downgrade

    with pytest.raises(PermissionDeniedError, match="manage permission required"):
        await storage.fs.rm(RESOURCE_FILE_URI, ctx=storage.ctx, no_vector=no_vector)

    storage.agfs.rm.assert_not_awaited()
    storage.vector.count.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)


@pytest.mark.parametrize("missing_after_lock", [False, True])
async def test_missing_resource_file_cleanup_rechecks_manage_after_stat(
    file_storage, missing_after_lock
):
    storage = file_storage
    storage.acl_level = AclLevel.MANAGE

    async def stat_with_acl_downgrade(path):
        if missing_after_lock and storage.agfs.stat.await_count == 1:
            return {"isDir": False}
        storage.acl_level = AclLevel.WRITE
        raise FileNotFoundError(path)

    storage.agfs.stat.side_effect = stat_with_acl_downgrade

    with pytest.raises(PermissionDeniedError, match="manage permission required"):
        await storage.fs.rm(RESOURCE_FILE_URI, ctx=storage.ctx)

    storage.agfs.rm.assert_not_awaited()
    storage.vector.count.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()
    if missing_after_lock:
        storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)
    else:
        storage.agfs.pathlock_acquire_exact.assert_not_awaited()


@pytest.mark.parametrize("no_vector", [False, True])
async def test_resource_file_missing_after_lock_cleans_orphans_with_manage(file_storage, no_vector):
    storage = file_storage
    storage.acl_level = AclLevel.MANAGE
    storage.agfs.stat.side_effect = [{"isDir": False}, FileNotFoundError(RESOURCE_FILE_PATH)]

    result = await storage.fs.rm(RESOURCE_FILE_URI, ctx=storage.ctx, no_vector=no_vector)

    assert result == {"estimated_deleted_count": 0 if no_vector else 7}
    storage.agfs.rm.assert_not_awaited()
    storage.agfs.pathlock_release.assert_awaited_once_with(storage.lease)
    if no_vector:
        storage.vector.count.assert_not_awaited()
        storage.vector.delete_uris.assert_not_awaited()
    else:
        storage.vector.delete_uris.assert_awaited_once_with(storage.ctx, [RESOURCE_FILE_URI])


async def test_resource_remove_files_preserves_borrowed_lease_and_skips_vectors(file_storage):
    storage = file_storage
    storage.acl_level = AclLevel.MANAGE
    outer_lease = {"lease_ref": "outer-lease"}

    await storage.fs.remove_files(RESOURCE_FILE_URI, ctx=storage.ctx, lease_ref=outer_lease)

    storage.agfs.rm.assert_awaited_once_with(
        RESOURCE_FILE_PATH,
        recursive=False,
        fs_ctx={"account_id": "account", "lease_ref": "outer-lease"},
        auto_pathlock=True,
    )
    storage.agfs.pathlock_acquire_exact.assert_not_awaited()
    storage.agfs.pathlock_release.assert_not_awaited()
    storage.vector.count.assert_not_awaited()
    storage.vector.delete_uris.assert_not_awaited()
