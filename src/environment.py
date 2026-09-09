"""Environment capture and verification.

The reproducibility claim in the README is that deleting `data/processed/` and
re-running the experiment returns every table byte-identical. That claim is only
true for a given set of library versions: `requirements.txt` carries lower bounds,
so a fresh install a year from now may resolve to a pandas or statsmodels release
that shifts a number without anything appearing to break.

This module makes the environment part of the recorded result. The versions the
published tables were produced under are pinned below; `check()` compares them
against what is actually installed and says so loudly when they differ.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

# The environment that produced the committed result tables. Update this only
# alongside a deliberate re-run that regenerates every table.
REFERENCE = {
    "python": "3.12.14",
    "numpy": "2.5.3",
    "pandas": "3.0.5",
    "scipy": "1.18.1",
    "statsmodels": "0.15.0",
    "scikit-learn": "1.9.0",
    "matplotlib": "3.11.1",
}


def current() -> dict[str, str]:
    """Versions actually installed right now."""
    out = {"python": platform.python_version()}
    for pkg in REFERENCE:
        if pkg == "python":
            continue
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "MISSING"
    return out


def as_frame() -> pd.DataFrame:
    """Recorded vs installed, as a table suitable for committing."""
    now = current()
    return pd.DataFrame([
        {"package": pkg, "recorded": REFERENCE[pkg], "installed": now[pkg],
         "matches": REFERENCE[pkg] == now[pkg]}
        for pkg in REFERENCE
    ])


def check(verbose: bool = True) -> bool:
    """True if the installed environment matches the one that produced the tables.

    Does not raise. A mismatch is not an error — newer libraries are usually fine
    — but it does mean a byte-identical reproduction is no longer guaranteed, and
    the reader deserves to be told rather than left to wonder why a digit moved.
    """
    frame = as_frame()
    mismatched = frame[~frame["matches"]]
    if verbose and len(mismatched):
        print("\n  ! Environment differs from the one that produced the committed tables.")
        print("    Byte-identical reproduction is not guaranteed. Install from")
        print("    requirements-lock.txt to reproduce the published numbers exactly.\n")
        for _, r in mismatched.iterrows():
            print(f"      {r['package']:<14} recorded {r['recorded']:<10} installed {r['installed']}")
        print()
    elif verbose:
        print(f"  environment matches the recorded one "
              f"(python {frame.loc[0, 'installed']}, pandas {current()['pandas']})")
    return bool(len(mismatched) == 0)


if __name__ == "__main__":
    ok = check()
    print(as_frame().to_string(index=False))
    sys.exit(0 if ok else 1)
