"""Scan current Git candidates without printing matching credential values.

This deliberately small baseline catches recognizable keys/private keys and
forbidden local configuration paths. It is not a historical or universal secret
detector; incident review and provider-side rotation remain operator tasks.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_PATTERNS = {
    "ally-api-key": re.compile(r"\bally_[A-Za-z0-9_-]{30,}"),
    "provider-api-key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    category: str


def scan_text(path: str, source: str) -> list[Finding]:
    """Return locations and categories only; never retain the matched value."""
    findings = []
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if (
        normalized == "local_agent/config.json"
        or (name == ".env" or name.startswith(".env."))
        and name != ".env.example"
        or name.endswith((".secret", ".secrets", ".dpapi"))
    ):
        findings.append(Finding(path, 1, "local-secret-file"))
    for number, line in enumerate(source.splitlines(), 1):
        for category, pattern in _PATTERNS.items():
            if pattern.search(line):
                findings.append(Finding(path, number, category))
    return findings


def scan_repository(root: Path) -> list[Finding]:
    """Scan tracked and non-ignored untracked files, including staged secrets."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    findings = []
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        path = root / name
        if path.is_symlink() or not path.is_file():
            continue
        source = path.read_bytes().decode("utf-8", errors="replace")
        findings.extend(scan_text(name, source))
    # The index can still contain a credential after the worktree was cleaned.
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACM"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    for name in set(staged.stdout.decode("utf-8").split("\0")) - {""}:
        blob = subprocess.run(
            ["git", "show", f":{name}"],
            cwd=root,
            capture_output=True,
            check=True,
        )
        findings.extend(scan_text(name, blob.stdout.decode("utf-8", errors="replace")))
    return sorted(set(findings), key=lambda finding: (finding.path, finding.line, finding.category))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan_repository(root)
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.category} (value redacted)")
    print(f"Secret scan: {len(findings)} finding(s); current Git candidates only.")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
