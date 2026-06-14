#!/usr/bin/env python3
"""
Pre-commit hook for validating site data files.

This script is called by git before each commit (if git config core.hooksPath is set).
If validation fails, the commit is blocked (exit code 1).

To enable:
  git config core.hooksPath scripts

To bypass (if needed):
  git commit --no-verify

Exit codes:
  0 = validation passed, commit allowed
  1 = validation failed, commit blocked
"""

import sys
import subprocess
from pathlib import Path


def main():
    print("[Validate site data before commit]")

    # Call the main validator script
    repo_root = Path(__file__).parent.parent
    validator_script = repo_root / "scripts" / "validate_site_data.py"
    docs_dir = repo_root / "docs"

    result = subprocess.run(
        [sys.executable, str(validator_script), "--docs-dir", str(docs_dir)],
        cwd=repo_root
    )

    if result.returncode != 0:
        print("\nCommit blocked: site data validation failed")
        print("Fix the errors above and try again, or use 'git commit --no-verify' to bypass")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
