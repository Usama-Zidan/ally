"""Pair this Windows companion using a secret from the backend operator."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from device_credentials import (
    DeviceCredentials,
    credentials_path,
    ensure_storage_available,
    save_credentials,
    validate_server_url,
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        # An enrollment secret must never be forwarded to another origin.
        return None


def redeem(server_url: str, token: str) -> DeviceCredentials:
    url = validate_server_url(server_url)
    if not 32 <= len(token) <= 128:
        raise ValueError("Invalid pairing token")
    parsed = urlsplit(url)
    endpoint = urlunsplit(
        (
            "https" if parsed.scheme == "wss" else "http",
            parsed.netloc,
            "/auth/devices/redeem",
            "",
            "",
        )
    )
    request = Request(
        endpoint,
        data=json.dumps({"token": token}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(_NoRedirect()).open(request, timeout=10) as response:
        data = response.read(4097)
    if len(data) > 4096:
        raise ValueError("Pairing response exceeds size limit")
    payload = json.loads(data)
    return DeviceCredentials(url, payload["device_id"], payload["api_key"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        if not sys.stdin.isatty():
            raise RuntimeError("Enrollment requires an interactive local terminal")
        with open(args.config_path, encoding="utf-8") as stream:
            config = json.load(stream)
        server_url = validate_server_url(config["server_url"])
        path = credentials_path(args.config_path, config)
        if path.exists() and not args.replace:
            raise FileExistsError("Device credential exists; use --replace for explicit re-pairing")
        ensure_storage_available()
        print(f"Pair with {server_url}. This replaces this tenant's current device.")
        if input("Type PAIR to approve enrollment on this PC: ") != "PAIR":
            raise ValueError("Enrollment declined locally")
        token = getpass.getpass("One-time pairing token: ")
        enrollment = redeem(server_url, token)
        save_credentials(path, enrollment, replace=args.replace)
        print("Device paired. Its credential is protected for this Windows user.")
    except Exception as exc:
        # Never include raw responses, tokens or credentials in terminal errors.
        print(f"Pairing failed ({type(exc).__name__}). Request a new token before retrying.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
