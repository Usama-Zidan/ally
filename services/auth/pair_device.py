"""Operator CLI: issue a one-time companion pairing token, or revoke enrollment."""

from __future__ import annotations

import argparse

from services.auth.devices import PAIRING_TTL_SECONDS, DeviceRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("issue", "revoke"))
    parser.add_argument("tenant_id")
    args = parser.parse_args()
    registry = DeviceRegistry()
    if args.command == "issue":
        token = registry.issue_pairing(args.tenant_id)
        print(f"Pairing token (expires in {PAIRING_TTL_SECONDS}s): {token}")
        print("Enter it only in the intended companion's local pairing prompt.")
        print("Redemption replaces the tenant's current companion and revokes its credential.")
    else:
        registry.revoke(args.tenant_id)
        print("Device enrollment and pending pairing token revoked.")


if __name__ == "__main__":
    main()
