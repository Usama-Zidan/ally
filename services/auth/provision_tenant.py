"""
Admin CLI for provisioning tenants and API keys.

Deliberately not exposed as an HTTP endpoint — anyone who could call a
"create tenant" API could mint themselves valid credentials. Run this
directly against the database instead:

    python -m services.auth.provision_tenant "Acme Corp"

Prints the raw API key exactly once. Only its hash is stored — if it's
lost, revoke it and provision a new one rather than trying to recover it.
"""
from __future__ import annotations

import argparse

from services.auth.api_keys import create_tenant_with_key, revoke_api_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a tenant and issue it an API key")
    create_parser.add_argument("tenant_name")

    revoke_parser = subparsers.add_parser("revoke", help="Revoke an existing API key")
    revoke_parser.add_argument("api_key")

    args = parser.parse_args()

    if args.command == "create":
        tenant_id, raw_key = create_tenant_with_key(args.tenant_name)
        print(f"tenant_id: {tenant_id}")
        print(f"api_key:   {raw_key}")
        print("\nStore this key now — it will not be shown again.")
    elif args.command == "revoke":
        revoked = revoke_api_key(args.api_key)
        print("Key revoked." if revoked else "No matching active key found.")


if __name__ == "__main__":
    main()
