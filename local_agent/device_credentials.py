"""Windows current-user DPAPI storage for a server-bound companion credential."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID


@dataclass(frozen=True)
class DeviceCredentials:
    server_url: str
    device_id: str
    api_key: str = field(repr=False)


def validate_server_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"ws", "wss"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/ws/pc-agent"
        or parsed.scheme == "ws"
        and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("Use wss://HOST/ws/pc-agent (ws:// is permitted only on loopback)")
    # Accessing .port also rejects malformed/out-of-range ports.
    _ = parsed.port
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path, "", ""))


def credentials_path(config_path: str, config: dict) -> Path:
    filename = config.get("device_credentials_file", "device-credentials.dpapi")
    if not isinstance(filename, str) or not filename:
        raise ValueError("Invalid device credential filename")
    path = Path(filename)
    if path.suffix.lower() != ".dpapi":
        raise ValueError("Device credential filename must end in .dpapi")
    return path if path.is_absolute() else Path(config_path).resolve().parent / path


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _protect(data: bytes, *, decrypt: bool = False) -> bytes:
    if sys.platform != "win32":
        raise RuntimeError("Companion credential storage requires Windows current-user DPAPI")
    if os.name != "nt":
        raise RuntimeError("Companion credential storage requires Windows current-user DPAPI")
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_Blob),
        ctypes.c_void_p,
        ctypes.POINTER(_Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_Blob),
    ]
    function.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    incoming = _Blob(len(data), buffer)
    outgoing = _Blob()
    # CRYPTPROTECT_UI_FORBIDDEN; never machine-wide protection or a plaintext fallback.
    if not function(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise OSError("Windows credential protection failed")
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        kernel.LocalFree(outgoing.data)


def ensure_storage_available() -> None:
    # Verify storage before redeeming a single-use token.
    probe = b"ally-device-storage-check"
    if _protect(_protect(probe), decrypt=True) != probe:
        raise RuntimeError("Windows credential protection is unavailable")


def save_credentials(path: Path, credentials: DeviceCredentials, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(
            "A device credential already exists; explicit replacement is required"
        )
    validate_server_url(credentials.server_url)
    UUID(credentials.device_id)
    if not credentials.api_key or len(credentials.api_key) > 256:
        raise ValueError("Invalid device credential")
    payload = json.dumps(
        {
            "version": 1,
            "server_url": credentials.server_url,
            "device_id": credentials.device_id,
            "api_key": credentials.api_key,
        }
    ).encode("utf-8")
    protected = _protect(payload)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(protected)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # Hard-link creation is atomic and refuses a concurrent overwrite.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_credentials(path: Path, server_url: str) -> DeviceCredentials:
    expected_url = validate_server_url(server_url)
    with path.open("rb") as stream:
        data = stream.read(16385)
    if len(data) > 16384:
        raise ValueError("Device credential file exceeds size limit")
    payload = json.loads(_protect(data, decrypt=True))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid device credential file")
    if validate_server_url(payload["server_url"]) != expected_url:
        raise ValueError("Device credential belongs to another server; pair explicitly")
    device_id = str(UUID(payload["device_id"]))
    api_key = payload["api_key"]
    if not isinstance(api_key, str) or not api_key or len(api_key) > 256:
        raise ValueError("Invalid device credential file")
    return DeviceCredentials(expected_url, device_id, api_key)
