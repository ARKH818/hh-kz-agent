"""Reusable, platform-specific protection for application-owned files.

Phase 5.7 protects browser profiles, SQLite files, logs and configuration
files without moving, renaming or rewriting them.  Every public helper fails
closed: a permission error is raised instead of being reported as success.
"""

from __future__ import annotations

import ctypes
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, unquote, urlsplit

SECURE_DIR_MODE = 0o700
SECURE_FILE_MODE = 0o600

# Well-known SIDs kept intentionally: SYSTEM and the local Administrators
# group must retain access on Windows.  Every other trustee is removed.
_WINDOWS_SYSTEM_SID = "S-1-5-18"
_WINDOWS_ADMINISTRATORS_SID = "S-1-5-32-544"

# Win32 constants used by the ACL implementation.
_FILE_ALL_ACCESS = 0x001F01FF
_GRANT_ACCESS = 1
_OBJECT_INHERIT_ACE = 0x01
_CONTAINER_INHERIT_ACE = 0x02
_SE_FILE_OBJECT = 1
_DACL_SECURITY_INFORMATION = 0x04
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_TRUSTEE_IS_SID = 0
_TRUSTEE_IS_UNKNOWN = 0
_ERROR_SUCCESS = 0

# Windows reparse points include symlinks, junctions and mount points.  Any of
# them can redirect a later file operation away from the validated path.
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
# GetFileAttributesW uses these errors for components that simply do not exist.
# Any other failure means the path could not be verified and must fail closed.
_ERROR_FILE_NOT_FOUND = 0x02
_ERROR_PATH_NOT_FOUND = 0x03

# Win32/NT constants for handle-relative opens.  FILE_OPEN_REPARSE_POINT makes
# the open land on the link itself, so the handle can be inspected before any
# read/write operation follows the redirect.
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_READ_ATTRIBUTES = 0x00000080
_READ_CONTROL = 0x00020000
_WRITE_DAC = 0x00040000
_SYNCHRONIZE = 0x00100000
_FILE_GENERIC_READ = 0x00120089
_FILE_GENERIC_WRITE = 0x00120116
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_OPEN = 1
_FILE_CREATE = 2
_FILE_OPEN_IF = 3
_FILE_DIRECTORY_FILE = 0x00000001
_FILE_NON_DIRECTORY_FILE = 0x00000040
_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_FILE_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_OBJ_CASE_INSENSITIVE = 0x00000040
_STATUS_OBJECT_NAME_NOT_FOUND = 0xC0000034
_STATUS_OBJECT_NAME_COLLISION = 0xC0000035
_STATUS_FILE_IS_A_DIRECTORY = 0xC00000BA

_SQLITE_URI_PREFIX = "file:"
_SQLITE_DRIVE_IN_URI = re.compile(r"^/([A-Za-z]:[\\/].*)$")


class FilesystemProtectionError(RuntimeError):
    """A sensitive path could not be secured."""


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", ctypes.c_ulong)]


class _TokenUser(ctypes.Structure):
    _fields_ = [("User", _SID_AND_ATTRIBUTES)]


class _TRUSTEE_W(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_ulong),
        ("TrusteeForm", ctypes.c_ulong),
        ("TrusteeType", ctypes.c_ulong),
        ("ptstrName", ctypes.c_void_p),
    ]


class _EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", ctypes.c_ulong),
        ("grfAccessMode", ctypes.c_ulong),
        ("grfInheritance", ctypes.c_ulong),
        ("Trustee", _TRUSTEE_W),
    ]


if os.name == "nt":
    from ctypes import wintypes

    class _UNICODE_STRING(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", ctypes.c_wchar_p),
        ]

    class _OBJECT_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_UNICODE_STRING)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", ctypes.c_void_p),
            ("SecurityQualityOfService", ctypes.c_void_p),
        ]

    class _IO_STATUS_UNION(ctypes.Union):
        _fields_ = [("Status", ctypes.c_long), ("Pointer", ctypes.c_void_p)]

    class _IO_STATUS_BLOCK(ctypes.Structure):
        _fields_ = [
            ("StatusAndPointer", _IO_STATUS_UNION),
            ("Information", ctypes.c_void_p),
        ]

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]


@dataclass(frozen=True)
class SQLiteTarget:
    """A SQLite connection target together with its real filesystem path."""

    connect_name: str
    uri: bool
    path: Path | None
    in_memory: bool


def _lexical_absolute(path_text: str) -> Path:
    """Normalize a path without resolving symlinks."""
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(path_text))))


def _sqlite_uri_path(split) -> Path:
    if split.netloc and split.netloc.casefold() != "localhost":
        raise FilesystemProtectionError("SQLite URI authority is not supported")
    if split.fragment:
        raise FilesystemProtectionError("SQLite URI fragments are not supported")
    raw_path = unquote(split.path)
    if not raw_path:
        raise FilesystemProtectionError("SQLite URI does not contain a file path")
    if os.name == "nt":
        drive = _SQLITE_DRIVE_IN_URI.match(raw_path)
        if drive:
            raw_path = drive.group(1)
    _check_raw_components(raw_path, kind="SQLite path")
    return _lexical_absolute(raw_path)


def resolve_sqlite_target(value: str | Path) -> SQLiteTarget:
    """Classify and normalize a SQLite target before any connection is opened.

    Only ``:memory:`` and genuine in-memory ``file:`` URIs (``mode=memory`` or
    ``file::memory:``) are treated as non-filesystem databases.  Every other
    ``file:`` value is resolved to a concrete path and must be protected.
    """
    text = str(value)
    if text == ":memory:":
        return SQLiteTarget(":memory:", False, None, True)

    if text.startswith(_SQLITE_URI_PREFIX):
        split = urlsplit(text)
        query = parse_qs(split.query, keep_blank_values=True)
        mode = "".join(query.get("mode", ())).casefold()
        if mode == "memory" or split.path.casefold() == ":memory:":
            return SQLiteTarget(text, True, None, True)
        path = _sqlite_uri_path(split)
        # Keep the URI absolute: sqlite3 resolves a relative ``file:`` URI
        # against CWD at connect time, which would bypass the validated path.
        connect_name = path.as_uri() + (f"?{split.query}" if split.query else "")
        return SQLiteTarget(connect_name, True, path, False)

    if not text.strip():
        raise FilesystemProtectionError("empty SQLite path")
    _check_raw_components(text, kind="SQLite path")
    path = _lexical_absolute(text)
    return SQLiteTarget(str(path), False, path, False)


def _windows_file_attributes(path: Path) -> int | None:
    """Read Win32 file attributes without following the path."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetFileAttributesW.restype = wintypes.DWORD
    attributes = kernel32.GetFileAttributesW(str(path))
    if attributes == _INVALID_FILE_ATTRIBUTES:
        error = ctypes.get_last_error()
        if error in (_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND):
            return None
        raise FilesystemProtectionError(
            f"cannot verify path component (Win32 error {error}): {path}"
        )
    return int(attributes)


def _is_unsafe_component(path: Path) -> bool:
    """Detect symlinks and Windows reparse points without resolving them."""
    try:
        if path.is_symlink():
            return True
        # lstat() must succeed for an existing component; an unexpected error
        # means the component could not be verified and must fail closed.
        path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise FilesystemProtectionError(
            f"cannot verify path component: {path}"
        ) from exc
    if os.name == "nt":
        attributes = _windows_file_attributes(path)
        return attributes is not None and bool(
            attributes & _FILE_ATTRIBUTE_REPARSE_POINT
        )
    return False


def _reject_single_component(path: Path, *, kind: str) -> None:
    if _is_unsafe_component(path):
        raise FilesystemProtectionError(
            f"refusing {kind} with unsafe path component: {path}"
        )


def _reject_unsafe_components(path: Path, *, kind: str) -> None:
    """Reject unsafe components on the target and every existing parent."""
    for candidate in (path, *path.parents):
        _reject_single_component(candidate, kind=kind)


def _check_raw_components(path_text: str, *, kind: str) -> None:
    """Check components of the path as written, before lexical normalization.

    ``a/link/../file`` and ``a/link/./file`` must not lose a symlinked or
    junctioned component during ``normpath``.  Walking the original components
    with a logical stack lets us reject such a component before it is collapsed.
    """
    raw = Path(os.path.expanduser(path_text))
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    anchor = raw.anchor
    parts = list(raw.parts)
    if anchor and parts and parts[0] == anchor:
        parts = parts[1:]
    stack: list[str] = []
    for part in parts:
        if part in ("", os.curdir):
            continue
        if part == os.pardir:
            if stack:
                stack.pop()
            continue
        candidate = Path(anchor, *stack, part) if anchor else Path(*stack, part)
        _reject_single_component(candidate, kind=kind)
        stack.append(part)


def validate_path_before_io(path: Path | str, *, kind: str = "path") -> Path:
    """Validate a sensitive path before any read, write or directory creation.

    The final component may not exist yet.  Every existing component of the
    absolute path is checked for symlinks and Windows reparse points.  The path
    is normalized lexically (never via :meth:`Path.resolve`) and nothing is
    created, chmod-ed or otherwise modified by this function.

    Callers must invoke it *before* ``open``/``write_text``/``mkdir`` so a
    redirected path cannot reach an external file before protection fires.
    """
    _check_raw_components(str(path), kind=kind)
    target = _lexical_absolute(str(Path(path).expanduser()))
    _reject_unsafe_components(target, kind=kind)
    return target


def _open_posix_directory_handle(path: Path, *, kind: str) -> int:
    """Open every directory component with O_NOFOLLOW and keep the dirfd chain."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    anchor = Path(path.anchor or os.sep)
    try:
        current = os.open(anchor, flags)
    except OSError as exc:
        raise FilesystemProtectionError(
            f"cannot open root of {kind}: {type(exc).__name__}"
        ) from exc
    try:
        for part in path.relative_to(anchor).parts:
            try:
                child = os.open(part, flags, dir_fd=current)
            except OSError as exc:
                raise FilesystemProtectionError(
                    f"cannot open {kind} component {part}: {type(exc).__name__}"
                ) from exc
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _assert_windows_handle(
    handle: int,
    *,
    is_directory: bool,
    name: str,
    kernel32: ctypes.CDLL,
) -> None:
    """Verify an already opened handle without following its redirect."""
    info = _BY_HANDLE_FILE_INFORMATION()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
        error = ctypes.get_last_error()
        raise FilesystemProtectionError(
            f"cannot verify opened object {name} (Win32 error {error})"
        )
    attributes = int(info.dwFileAttributes)
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise FilesystemProtectionError(f"refusing reparse point: {name}")
    if bool(attributes & _FILE_ATTRIBUTE_DIRECTORY) is not is_directory:
        expected = "directory" if is_directory else "regular file"
        raise FilesystemProtectionError(f"{name} is not a {expected}")


def _windows_kernel32() -> ctypes.CDLL:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    ]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _windows_ntdll() -> ctypes.CDLL:
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtCreateFile.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_OBJECT_ATTRIBUTES),
        ctypes.POINTER(_IO_STATUS_BLOCK),
        ctypes.POINTER(wintypes.LARGE_INTEGER),
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        ctypes.c_void_p,
        wintypes.ULONG,
    ]
    ntdll.NtCreateFile.restype = ctypes.c_long
    return ntdll


def _raise_nt_status(status: int, name: str) -> None:
    status &= 0xFFFFFFFF
    if status == _STATUS_OBJECT_NAME_NOT_FOUND:
        raise FileNotFoundError(f"path component does not exist: {name}")
    if status == _STATUS_OBJECT_NAME_COLLISION:
        raise FileExistsError(f"path component already exists: {name}")
    if status == _STATUS_FILE_IS_A_DIRECTORY:
        raise FilesystemProtectionError(f"not a regular file: {name}")
    raise OSError(f"NtCreateFile failed for {name}: NTSTATUS 0x{status:08X}")


def _open_windows_anchor(path: Path, *, kind: str, kernel32: ctypes.CDLL) -> int:
    handle = kernel32.CreateFileW(
        str(path.anchor),
        _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        3,  # OPEN_EXISTING
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_OPEN_REPARSE_POINT,
        None,
    )
    if not handle or handle in (-1, 0xFFFFFFFFFFFFFFFF):
        raise OSError(
            f"cannot open root of {kind} (Win32 error {ctypes.get_last_error()})"
        )
    # Volume roots do not always support GetFileInformationByHandle, but their
    # attributes can be checked by path: the anchor is derived from the lexical
    # absolute path, not from a user-controllable relative component.
    attributes = _windows_file_attributes(Path(path.anchor))
    if attributes is None or attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        kernel32.CloseHandle(handle)
        raise FilesystemProtectionError(f"unsafe {kind} root: {path.anchor}")
    return int(handle)


def _nt_open_relative(
    root_handle: int,
    name: str,
    *,
    is_directory: bool,
    disposition: int,
    desired_access: int,
    kernel32: ctypes.CDLL,
    ntdll: ctypes.CDLL,
) -> int:
    from ctypes import wintypes

    text = str(name)
    unicode_name = _UNICODE_STRING(len(text) * 2, len(text) * 2, text)
    attributes = _OBJECT_ATTRIBUTES(
        ctypes.sizeof(_OBJECT_ATTRIBUTES),
        root_handle,
        ctypes.pointer(unicode_name),
        _OBJ_CASE_INSENSITIVE,
        None,
        None,
    )
    options = _FILE_OPEN_REPARSE_POINT | _FILE_SYNCHRONOUS_IO_NONALERT
    options |= _FILE_DIRECTORY_FILE if is_directory else _FILE_NON_DIRECTORY_FILE
    handle = wintypes.HANDLE()
    status_block = _IO_STATUS_BLOCK()
    status = ntdll.NtCreateFile(
        ctypes.byref(handle),
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(status_block),
        None,
        0,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        disposition,
        options,
        None,
        0,
    )
    if status:
        _raise_nt_status(status, name)
    handle_value = int(handle.value)
    try:
        _assert_windows_handle(
            handle_value, is_directory=is_directory, name=name, kernel32=kernel32
        )
    except BaseException:
        kernel32.CloseHandle(handle_value)
        raise
    return handle_value


def _walk_windows_directories(
    path: Path,
    *,
    kind: str,
    create_missing: bool,
    final_access: int | None = None,
) -> tuple[int, ctypes.CDLL, ctypes.CDLL]:
    """Open ``path`` component-by-component relative to a verified root handle.

    ``final_access`` is used for the last component when the caller needs to
    write its security descriptor.  Intermediate components keep the minimum
    attribute rights needed to open their children.
    """
    kernel32 = _windows_kernel32()
    ntdll = _windows_ntdll()
    current = _open_windows_anchor(path, kind=kind, kernel32=kernel32)
    parts = path.relative_to(Path(path.anchor)).parts
    try:
        for index, part in enumerate(parts):
            access = (
                final_access
                if final_access is not None and index == len(parts) - 1
                else _FILE_READ_ATTRIBUTES | _SYNCHRONIZE
            )
            try:
                child = _nt_open_relative(
                    current,
                    part,
                    is_directory=True,
                    disposition=_FILE_OPEN,
                    desired_access=access,
                    kernel32=kernel32,
                    ntdll=ntdll,
                )
            except FileNotFoundError:
                if not create_missing:
                    raise
                child = _nt_open_relative(
                    current,
                    part,
                    is_directory=True,
                    disposition=_FILE_CREATE,
                    desired_access=_FILE_READ_ATTRIBUTES
                    | _READ_CONTROL
                    | _WRITE_DAC
                    | _SYNCHRONIZE,
                    kernel32=kernel32,
                    ntdll=ntdll,
                )
                try:
                    _apply_windows_acl(child, is_directory=True)
                except BaseException:
                    kernel32.CloseHandle(child)
                    raise
            kernel32.CloseHandle(current)
            current = child
        return current, kernel32, ntdll
    except BaseException:
        kernel32.CloseHandle(current)
        raise


def _open_validated_file_posix(
    target: Path,
    mode: str,
    *,
    encoding: str | None,
    errors: str | None,
) -> object:
    reading = "r" in mode
    creating_exclusive = "x" in mode
    appending = "a" in mode
    flags = os.O_NOFOLLOW | os.O_CLOEXEC
    if reading:
        flags |= os.O_RDONLY
    else:
        flags |= os.O_WRONLY
        if creating_exclusive:
            flags |= os.O_CREAT | os.O_EXCL
        else:
            flags |= os.O_CREAT
            if appending:
                flags |= os.O_APPEND
            else:
                flags |= os.O_TRUNC

    parent_fd = _open_posix_directory_handle(target.parent, kind="validated file")
    try:
        fd = os.open(target.name, flags, SECURE_FILE_MODE, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise FilesystemProtectionError(f"not a regular file: {target.name}")
        if not reading:
            os.fchmod(fd, SECURE_FILE_MODE)
        if appending:
            os.lseek(fd, 0, os.SEEK_END)
        if "b" in mode:
            fd_mode = "rb" if reading else "ab" if appending else "wb"
        elif "+" in mode:
            fd_mode = "r+"
        elif reading:
            fd_mode = "r"
        elif appending:
            fd_mode = "a"
        else:
            fd_mode = "w"
        return os.fdopen(fd, fd_mode, encoding=encoding, errors=errors)
    except BaseException:
        os.close(fd)
        raise


def _open_validated_file_windows(
    target: Path,
    mode: str,
    *,
    encoding: str | None,
    errors: str | None,
) -> object:
    import msvcrt

    reading = "r" in mode
    creating_exclusive = "x" in mode
    appending = "a" in mode
    disposition = (
        _FILE_OPEN if reading else _FILE_CREATE if creating_exclusive else _FILE_OPEN_IF
    )
    access = (
        _FILE_GENERIC_READ
        if reading
        else _FILE_GENERIC_WRITE
        | _FILE_READ_ATTRIBUTES
        | _READ_CONTROL
        | _WRITE_DAC
        | _SYNCHRONIZE
    )
    parent, kernel32, ntdll = _walk_windows_directories(
        target.parent, kind="validated file", create_missing=False
    )
    try:
        handle = _nt_open_relative(
            parent,
            target.name,
            is_directory=False,
            disposition=disposition,
            desired_access=access,
            kernel32=kernel32,
            ntdll=ntdll,
        )
    finally:
        kernel32.CloseHandle(parent)
    try:
        if not reading:
            _apply_windows_acl(handle, is_directory=False)
        fd_flags = os.O_BINARY | (os.O_RDONLY if reading else os.O_WRONLY)
        fd = msvcrt.open_osfhandle(handle, fd_flags)
    except BaseException:
        kernel32.CloseHandle(handle)
        raise
    try:
        if appending:
            os.lseek(fd, 0, os.SEEK_END)
        elif "w" in mode or "x" in mode:
            os.ftruncate(fd, 0)
        if "b" in mode:
            fd_mode = "rb" if reading else "ab" if appending else "wb"
        elif "+" in mode:
            fd_mode = "r+"
        elif reading:
            fd_mode = "r"
        elif appending:
            fd_mode = "a"
        else:
            fd_mode = "w"
        return os.fdopen(fd, fd_mode, encoding=encoding, errors=errors)
    except BaseException:
        os.close(fd)
        raise


def open_validated_file(
    path: Path | str,
    mode: str,
    *,
    kind: str,
    encoding: str | None = None,
    errors: str | None = None,
) -> object:
    """Open a validated file through a no-follow/reparse-checked handle.

    On POSIX every parent component is opened relative to its predecessor with
    ``O_NOFOLLOW|O_DIRECTORY`` and the leaf with ``O_NOFOLLOW``.  On Windows
    every component is opened with ``NtCreateFile(RootDirectory=...)`` and
    ``FILE_OPEN_REPARSE_POINT``, then checked by handle.  The returned stream
    owns an already-verified object; later renames cannot redirect these writes.
    """
    target = validate_path_before_io(path, kind=kind)
    if os.name == "nt":
        return _open_validated_file_windows(
            target, mode, encoding=encoding, errors=errors
        )
    return _open_validated_file_posix(target, mode, encoding=encoding, errors=errors)


def read_validated_text(path: Path | str, *, kind: str) -> str | None:
    """Read a validated UTF-8 file by handle; return None when it is missing."""
    try:
        with open_validated_file(path, "r", kind=kind, encoding="utf-8") as stream:
            return stream.read()
    except FileNotFoundError:
        return None


def write_validated_text(path: Path | str, content: str, *, kind: str) -> None:
    """Create/overwrite a validated UTF-8 file with restrictive permissions."""
    with open_validated_file(path, "w", kind=kind, encoding="utf-8") as stream:
        stream.write(content)


def _ensure_validated_directory_posix(target: Path, *, kind: str) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    anchor = Path(target.anchor or os.sep)
    current = os.open(anchor, flags)
    try:
        for part in target.relative_to(anchor).parts:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(part, SECURE_DIR_MODE, dir_fd=current)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=current)
                os.fchmod(child, SECURE_DIR_MODE)
            except OSError as exc:
                raise FilesystemProtectionError(
                    f"cannot open {kind} component {part}: {type(exc).__name__}"
                ) from exc
            os.close(current)
            current = child
    finally:
        os.close(current)


def _ensure_validated_directory_windows(target: Path, *, kind: str) -> None:
    current, kernel32, ntdll = _walk_windows_directories(
        target, kind=kind, create_missing=True
    )
    kernel32.CloseHandle(current)


def ensure_validated_directory(path: Path | str, *, kind: str) -> Path:
    """Create a directory tree with handle-relative checks for each component.

    Every missing ancestor is created through the already-verified parent
    handle and restricted before the next component is opened.  Existing
    components are opened with no-follow/reparse-point semantics and are not
    otherwise modified.
    """
    target = validate_path_before_io(path, kind=kind)
    if os.name == "nt":
        _ensure_validated_directory_windows(target, kind=kind)
    else:
        _ensure_validated_directory_posix(target, kind=kind)
    return target


def validate_sqlite_target(target: SQLiteTarget) -> None:
    """Fail closed when a file-backed SQLite target is unsafe to open."""
    if target.in_memory or target.path is None:
        return
    path = validate_path_before_io(target.path, kind="SQLite path")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode):
        raise FilesystemProtectionError(
            f"SQLite target is not a regular file: {path.name}"
        )


def _validated_path(path: Path | str, *, kind: str) -> Path:
    target = validate_path_before_io(path, kind=kind)
    if not target.exists():
        raise FilesystemProtectionError(
            f"cannot secure missing {kind}: {target.name}"
        )
    return target


def _iter_tree(root: Path, *, recursive: bool) -> Iterator[tuple[Path, bool]]:
    """Yield ``(path, is_directory)`` without following links."""
    yield root, root.is_dir()
    if not recursive or not root.is_dir():
        return
    for directory, subdirectories, filenames in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        safe_subdirectories: list[str] = []
        for name in subdirectories:
            child = directory_path / name
            if _is_unsafe_component(child):
                continue
            safe_subdirectories.append(name)
            yield child, True
        subdirectories[:] = safe_subdirectories
        for name in filenames:
            child = directory_path / name
            if _is_unsafe_component(child):
                continue
            yield child, False


def _apply_posix(path: Path, *, is_directory: bool) -> None:
    mode = SECURE_DIR_MODE if is_directory else SECURE_FILE_MODE
    path.chmod(mode)


def _protect_posix(root: Path, *, recursive: bool) -> None:
    failures: list[str] = []
    for path, is_directory in _iter_tree(root, recursive=recursive):
        try:
            _apply_posix(path, is_directory=is_directory)
        except OSError as exc:
            failures.append(f"{path.name}:{type(exc).__name__}")
    if failures:
        raise FilesystemProtectionError(
            "POSIX permission update failed: " + ", ".join(failures)
        )


def _current_user_sid(advapi32: ctypes.CDLL, kernel32: ctypes.CDLL) -> str:
    """Return the SID of the account running this process."""
    from ctypes import wintypes

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HANDLE]
    kernel32.LocalFree.restype = wintypes.HANDLE

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
    ):
        raise FilesystemProtectionError("cannot open the current process token")
    try:
        length = wintypes.DWORD()
        advapi32.GetTokenInformation(
            token, _TOKEN_USER, None, 0, ctypes.byref(length)
        )
        buffer = (ctypes.c_byte * length.value)()
        if not advapi32.GetTokenInformation(
            token,
            _TOKEN_USER,
            ctypes.cast(buffer, ctypes.c_void_p),
            length.value,
            ctypes.byref(length),
        ):
            raise FilesystemProtectionError("cannot read the current user SID")
        token_user = ctypes.cast(buffer, ctypes.POINTER(_TokenUser)).contents
        sid_text = ctypes.c_wchar_p()
        if not advapi32.ConvertSidToStringSidW(
            token_user.User.Sid, ctypes.byref(sid_text)
        ):
            raise FilesystemProtectionError("cannot convert the current user SID")
        value = sid_text.value or ""
        kernel32.LocalFree(sid_text)
        if not value:
            raise FilesystemProtectionError("the current user SID is empty")
        return value
    finally:
        kernel32.CloseHandle(token)


def _apply_windows_acl(target: Path | int, *, is_directory: bool) -> None:
    """Replace the DACL with current-user + SYSTEM + Administrators only.

    An integer target is an already-opened handle.  Passing a handle lets the
    caller secure the exact object it verified, instead of re-opening a path
    that an attacker could replace in the meantime.
    """
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    advapi32.ConvertStringSidToSidW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
    advapi32.SetEntriesInAclW.argtypes = [
        wintypes.ULONG,
        ctypes.POINTER(_EXPLICIT_ACCESS_W),
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.SetEntriesInAclW.restype = wintypes.DWORD
    advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.HANDLE,
        ctypes.c_void_p,
    ]
    advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.SetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.HANDLE,
        ctypes.c_void_p,
    ]
    advapi32.SetSecurityInfo.restype = wintypes.DWORD

    sids = (
        _current_user_sid(advapi32, kernel32),
        _WINDOWS_SYSTEM_SID,
        _WINDOWS_ADMINISTRATORS_SID,
    )
    sid_pointers: list[ctypes.c_void_p] = []
    new_acl = wintypes.HANDLE()
    try:
        for sid in sids:
            pointer = ctypes.c_void_p()
            if not advapi32.ConvertStringSidToSidW(sid, ctypes.byref(pointer)):
                raise FilesystemProtectionError(f"cannot resolve SID {sid}")
            sid_pointers.append(pointer)

        inheritance = (
            _OBJECT_INHERIT_ACE | _CONTAINER_INHERIT_ACE if is_directory else 0
        )
        entries = (_EXPLICIT_ACCESS_W * len(sid_pointers))()
        for index, pointer in enumerate(sid_pointers):
            entries[index].grfAccessPermissions = _FILE_ALL_ACCESS
            entries[index].grfAccessMode = _GRANT_ACCESS
            entries[index].grfInheritance = inheritance
            entries[index].Trustee.TrusteeForm = _TRUSTEE_IS_SID
            entries[index].Trustee.TrusteeType = _TRUSTEE_IS_UNKNOWN
            entries[index].Trustee.ptstrName = pointer.value or 0

        result = advapi32.SetEntriesInAclW(
            len(entries), entries, None, ctypes.byref(new_acl)
        )
        if result != _ERROR_SUCCESS or not new_acl.value:
            raise FilesystemProtectionError("cannot build a restrictive DACL")

        flags = (
            _DACL_SECURITY_INFORMATION
            | _PROTECTED_DACL_SECURITY_INFORMATION
        )
        if isinstance(target, int):
            result = advapi32.SetSecurityInfo(
                target,
                _SE_FILE_OBJECT,
                flags,
                None,
                None,
                new_acl,
                None,
            )
        else:
            result = advapi32.SetNamedSecurityInfoW(
                str(target),
                _SE_FILE_OBJECT,
                flags,
                None,
                None,
                new_acl,
                None,
            )
        if result != _ERROR_SUCCESS:
            raise FilesystemProtectionError(
                f"cannot apply a restrictive DACL (Win32 error {result})"
            )
    finally:
        if new_acl.value:
            kernel32.LocalFree(new_acl)
        for pointer in sid_pointers:
            if pointer.value:
                kernel32.LocalFree(pointer)


def _protect_windows(root: Path, *, recursive: bool) -> None:
    failures: list[str] = []
    for path, is_directory in _iter_tree(root, recursive=recursive):
        try:
            _apply_windows_acl(path, is_directory=is_directory)
        except (FilesystemProtectionError, OSError) as exc:
            failures.append(f"{path.name}:{type(exc).__name__}")
    if failures:
        raise FilesystemProtectionError(
            "Windows ACL update failed: " + ", ".join(failures)
        )


def _restrict_existing_directory(target: Path) -> None:
    """Restrict the exact directory object that passed the preflight check."""
    if os.name == "nt":
        handle, kernel32, ntdll = _walk_windows_directories(
            target,
            kind="directory",
            create_missing=False,
            final_access=_FILE_READ_ATTRIBUTES
            | _READ_CONTROL
            | _WRITE_DAC
            | _SYNCHRONIZE,
        )
        try:
            _apply_windows_acl(handle, is_directory=True)
        finally:
            kernel32.CloseHandle(handle)
        return

    fd = _open_posix_directory_handle(target, kind="directory")
    try:
        os.fchmod(fd, SECURE_DIR_MODE)
    finally:
        os.close(fd)


def _restrict_existing_file(target: Path) -> None:
    """Restrict the exact file object that passed the preflight check."""
    if os.name == "nt":
        parent, kernel32, ntdll = _walk_windows_directories(
            target.parent, kind="file", create_missing=False
        )
        try:
            handle = _nt_open_relative(
                parent,
                target.name,
                is_directory=False,
                disposition=_FILE_OPEN,
                desired_access=_FILE_READ_ATTRIBUTES
                | _READ_CONTROL
                | _WRITE_DAC
                | _SYNCHRONIZE,
                kernel32=kernel32,
                ntdll=ntdll,
            )
        finally:
            kernel32.CloseHandle(parent)
        try:
            _apply_windows_acl(handle, is_directory=False)
        finally:
            kernel32.CloseHandle(handle)
        return

    parent_fd = _open_posix_directory_handle(target.parent, kind="file")
    try:
        fd = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise FilesystemProtectionError(f"not a regular file: {target.name}")
        os.fchmod(fd, SECURE_FILE_MODE)
    finally:
        os.close(fd)


def protect_directory(path: Path | str, *, recursive: bool = True) -> None:
    """Restrict a directory (and optionally its existing contents)."""
    target = _validated_path(path, kind="directory")
    if not target.is_dir():
        raise FilesystemProtectionError(
            f"not a directory: {target.name}"
        )
    if not recursive:
        _restrict_existing_directory(target)
        return
    if os.name == "nt":
        _protect_windows(target, recursive=True)
    else:
        _protect_posix(target, recursive=True)


def protect_file(path: Path | str) -> None:
    """Restrict a single regular file without following its final link."""
    target = _validated_path(path, kind="file")
    _restrict_existing_file(target)


def protect_config_file(path: Path | str) -> None:
    """Restrict a configuration file containing tokens or personal data."""
    protect_file(path)


def protect_log_file(path: Path | str) -> None:
    """Restrict a diagnostic log file and its containing directory."""
    # Validate even when the file does not exist yet: a symlinked or junctioned
    # parent must be rejected before the log handler creates the file there.
    target = validate_path_before_io(path, kind="log file")
    if target.parent.exists():
        protect_directory(target.parent, recursive=False)
    if target.exists():
        protect_file(target)


_SQLITE_SIDE_SUFFIXES = ("-journal", "-wal", "-shm", "-sqlite", "-tmp")


def protect_sqlite_siblings(path: Path | str) -> None:
    """Protect the database, its directory and all existing SQLite side files.

    Side files are scanned with ``DirEntry.stat(follow_symlinks=False)``.  A
    symlink, junction or other reparse point with a SQLite side-file name is
    rejected instead of being silently followed or protected.  Every object
    that is protected is then re-opened by handle in :func:`protect_file`.
    """
    db_path = validate_path_before_io(path, kind="SQLite path")
    parent = validate_path_before_io(db_path.parent, kind="SQLite directory")
    protect_directory(parent, recursive=False)
    candidates: list[Path] = []
    try:
        info = db_path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise FilesystemProtectionError(
            f"cannot verify SQLite database: {type(exc).__name__}"
        ) from exc
    else:
        attributes = getattr(info, "st_file_attributes", 0)
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT or not stat.S_ISREG(
            info.st_mode
        ):
            raise FilesystemProtectionError(
                f"unsafe SQLite database: {db_path.name}"
            )
        candidates.append(db_path)

    try:
        with os.scandir(parent) as entries:
            for entry in entries:
                if not entry.name.startswith(db_path.name):
                    continue
                suffix = entry.name[len(db_path.name) :]
                if not suffix or not suffix.startswith(_SQLITE_SIDE_SUFFIXES):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise FilesystemProtectionError(
                        f"cannot verify SQLite side file {entry.name}: "
                        f"{type(exc).__name__}"
                    ) from exc
                attributes = getattr(info, "st_file_attributes", 0)
                if (
                    entry.is_symlink()
                    or attributes & _FILE_ATTRIBUTE_REPARSE_POINT
                    or not stat.S_ISREG(info.st_mode)
                ):
                    raise FilesystemProtectionError(
                        f"unsafe SQLite side file: {entry.name}"
                    )
                candidates.append(Path(entry.path))
    except FilesystemProtectionError:
        raise
    except OSError as exc:
        raise FilesystemProtectionError(
            f"cannot scan SQLite directory: {type(exc).__name__}"
        ) from exc

    for candidate in candidates:
        protect_file(candidate)


def protect_database(path: Path | str | SQLiteTarget) -> None:
    """Protect the SQLite directory, database and existing journal files.

    Each missing parent is validated immediately before creation and protected
    before the next level is created.  A file-backed target is validated again
    after directory creation so a symlink cannot redirect SQLite outside the
    protected tree.
    """
    target = path if isinstance(path, SQLiteTarget) else resolve_sqlite_target(path)
    validate_sqlite_target(target)
    if target.in_memory or target.path is None:
        return

    db_path = target.path
    ensure_validated_directory(db_path.parent, kind="SQLite directory")
    validate_sqlite_target(target)
    protect_sqlite_siblings(db_path)


def protect_browser_profile(path: Path | str) -> None:
    """Protect an existing persistent browser profile without modifying it."""
    protect_directory(path, recursive=True)
