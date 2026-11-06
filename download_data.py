"""Download, verify and safely extract the versioned public data release.

Uses only Python's standard library. No Git LFS, account or API token needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile

REPOSITORY = "bastenefg/biomaterials-atlas"
TAG = "data-2026-11-06"
ASSET = "biomaterials-atlas-data-2026-11-06.zip"
BASE = f"https://github.com/{REPOSITORY}/releases/download/{TAG}"
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXTRACTED_BYTES = 2 * 1024**3
CHUNK = 1024**2


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            result.update(chunk)
    return result.hexdigest()


def relative_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError(f"Unsafe path in archive or manifest: {name!r}")
    return path


def verify_data(directory: Path) -> dict:
    """Verify each payload byte against the checksum-protected manifest."""
    directory = directory.resolve()
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = set()
    for entry in manifest["files"]:
        relative = relative_path(entry["path"])
        candidate = directory / relative
        if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents if parent != directory and parent.is_relative_to(directory)):
            raise ValueError(f"Symlink inside the data directory: {relative}")
        path = candidate.resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"File escapes the data directory: {relative}")
        if str(relative) in expected:
            raise ValueError(f"Duplicate manifest path: {relative}")
        expected.add(str(relative))
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Checksum or byte-size mismatch: {relative}")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != expected | {"manifest.json"}:
        raise ValueError("Extracted file set differs from the manifest")
    return manifest


def safe_extract(archive: Path, staging: Path) -> Path:
    """Extract only checked regular data/ members into a new empty directory."""
    staging = staging.resolve()
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        if len(members) > 5000 or sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
            raise ValueError("Archive exceeds release file-count or size limits")
        seen = set()
        for member in members:
            name = member.filename.rstrip("/") if member.is_dir() else member.filename
            relative = relative_path(name)
            if relative.parts[0] != "data" or len(relative.parts) < 2 and not member.is_dir():
                raise ValueError("Archive must contain only a data/ directory")
            key = str(relative).casefold()
            if key in seen:
                raise ValueError(f"Duplicate archive path: {relative}")
            seen.add(key)
            mode = (member.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise ValueError(f"Archive contains a nonregular entry: {relative}")
            if member.flag_bits & 0x1:
                raise ValueError("Encrypted archive entries are unsupported")
            target = (staging / relative).resolve()
            if not target.is_relative_to(staging):
                raise ValueError(f"Archive path escapes output: {relative}")
        required = sum(member.file_size for member in members) + 256 * 1024**2
        if shutil.disk_usage(staging).free < required:
            raise OSError("Not enough disk space to extract the data and keep a 256 MiB reserve")
        for member in members:
            target = staging / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, CHUNK)
    data = staging / "data"
    verify_data(data)
    return data


def fetch_checksum() -> str:
    request = urllib.request.Request(f"{BASE}/SHA256SUMS", headers={"User-Agent": "biomaterials-atlas-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read(65537)
    if len(text) > 65536:
        raise ValueError("Unexpectedly large checksum file")
    for line in text.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if match and match.group(2) == ASSET:
            return match.group(1).lower()
    raise ValueError(f"No checksum found for {ASSET}")


def download(destination: Path, expected_sha256: str) -> None:
    request = urllib.request.Request(f"{BASE}/{ASSET}", headers={"User-Agent": "biomaterials-atlas-downloader"})
    result = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("xb") as output:
        declared = int(response.headers.get("Content-Length", "0"))
        if declared > MAX_ARCHIVE_BYTES:
            raise ValueError("Release download exceeds the archive size limit")
        if declared and shutil.disk_usage(destination.parent).free < declared + 256 * 1024**2:
            raise OSError("Not enough disk space for the download")
        while chunk := response.read(CHUNK):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError("Release download exceeds the archive size limit")
            result.update(chunk)
            output.write(chunk)
    if result.hexdigest() != expected_sha256:
        raise ValueError("Downloaded archive SHA-256 does not match SHA256SUMS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("."), help="Parent directory for the new data/ folder")
    parser.add_argument("--verify-only", type=Path, metavar="DATA_DIR", help="Verify an already extracted data directory")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_data(args.verify_only)
        print(f"Verified {len(manifest['files'])} payload files.")
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "data"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; choose a different --output directory")
    expected = fetch_checksum()
    with tempfile.TemporaryDirectory(prefix=".atlas-download-", dir=output) as temporary:
        staging = Path(temporary)
        archive = staging / ASSET
        print(f"Downloading {TAG}…", flush=True)
        download(archive, expected)
        print("Archive checksum verified. Extracting and checking payload files…", flush=True)
        data = safe_extract(archive, staging)
        data.rename(target)
    print(f"Ready: {target}")


if __name__ == "__main__":
    main()
