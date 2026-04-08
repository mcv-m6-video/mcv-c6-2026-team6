#!/usr/bin/env python3
"""
Extract SN-BAS-2025 train.zip / valid.zip / test.zip into one output directory.
These archives use WinZip-style AES encryption. GNU unzip often skips all payload
files even with -P (empty match folders). This uses pyzipper, or optionally 7z.

Usage:
    python extract_split_zips.py <zip_dir> <out_dir> <password>

    zip_dir   — directory containing train.zip, valid.zip, test.zip
    out_dir   — destination for the merged england_efl/... tree
    password  — SoccerNet NDA password (WinZip AES)
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import os


def extract_pyzipper(zip_path: str, out_dir: str, password: str) -> None:
    import pyzipper
    print(f"  pyzipper: extracting {os.path.basename(zip_path)} ...", flush=True)
    with pyzipper.AESZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir, pwd=password.encode())


def extract_7z(zip_path: str, out_dir: str, password: str) -> None:
    print(f"  7z: extracting {os.path.basename(zip_path)} ...", flush=True)
    result = subprocess.run(
        ["7z", "x", f"-p{password}", f"-o{out_dir}", "-y", zip_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"7z failed for {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_dir", help="Directory with train/valid/test .zip files")
    parser.add_argument("out_dir", help="Output directory for merged england_efl/... tree")
    parser.add_argument("password", help="SoccerNet NDA zip password")
    parser.add_argument(
        "--backend", choices=["auto", "pyzipper", "7z"], default="auto",
        help="Extraction backend (default: auto — pyzipper preferred, 7z fallback)"
    )
    args = parser.parse_args()

    # Decide backend
    use_pyzipper = False
    use_7z = False
    if args.backend in ("auto", "pyzipper"):
        try:
            import pyzipper  # noqa: F401
            use_pyzipper = True
        except ImportError:
            if shutil.which("7z"):
                use_7z = True
            else:
                print("ERROR: pyzipper not installed and 7z not found.", file=sys.stderr)
                print("  pip install pyzipper   OR   apt install p7zip-full", file=sys.stderr)
                sys.exit(1)
    elif args.backend == "7z":
        if not shutil.which("7z"):
            print("ERROR: 7z not found on PATH.", file=sys.stderr)
            sys.exit(1)
        use_7z = True

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("train", "valid", "test"):
        zip_path = os.path.join(args.zip_dir, f"{split}.zip")
        if not os.path.isfile(zip_path):
            print(f"WARNING: {zip_path} not found — skipping.", file=sys.stderr)
            continue
        if use_pyzipper:
            extract_pyzipper(zip_path, args.out_dir, args.password)
        elif use_7z:
            extract_7z(zip_path, args.out_dir, args.password)

    print("Done.")


if __name__ == "__main__":
    main()
