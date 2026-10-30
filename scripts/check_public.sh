#!/usr/bin/env bash
# Fail if the public tree contains local paths, credentials, or saved downloads.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - "$root" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
# Pieces stay split so this checker does not itself contain the banned strings.
needles = (
    "/Users" + "/",
    "/home" + "/",
    "ana" + "conda",
    "Franklin" + "ITC",
    "washpost" + ".com",
    "AK" + "IA",
    "github_pat" + "_",
    "BEGIN " + "PRIVATE KEY",
    "AWS_" + "SECRET",
    "nollb" + "/",
    "AWS_PROFILE=" + "bennollweather",
)
text_suffixes = {".py", ".md", ".yml", ".yaml", ".html", ".sh", ".txt", ".env", ".gitignore"}
bad = []
for path in root.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    if path.suffix.lower() in {".grib", ".grib2", ".nc", ".npz", ".zarr"}:
        bad.append(f"{path.relative_to(root)} is a data file")
        continue
    if path.suffix.lower() not in text_suffixes and path.name not in {"requirements.txt"}:
        continue
    text = path.read_text(errors="ignore")
    for needle in needles:
        if needle in text:
            bad.append(f"{path.relative_to(root)} contains {needle}")
if (root / "config" / "settings.local.env").exists():
    bad.append("config/settings.local.env must not be committed")
if bad:
    print("\n".join(bad))
    sys.exit(1)
print("Public tree has no local paths, credentials, or saved downloads.")
PY
