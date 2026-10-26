#!/usr/bin/env python3
"""Build and invoke the M4 trainer without writing into the Bullet checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib


PINNED_BULLET_COMMIT = "579b78b37a694cacd6e9ede4706386c15c34837c"
PINNED_BULLET_PATCH_SHA256 = "ecf8ce81486d5e2d3cfbd881bbd053e8b3682b8de4acf24a6824310cb29b5141"
FORBIDDEN_LOADER_NAMES = ("bullet-utils", "montyformat", "sfbinpack", "viriformat", "montybinpack", "viribinpack", "bulletformat")
ROOT = Path(__file__).resolve().parents[2]
DRIVER_SOURCE = ROOT / "tools" / "train" / "driver" / "src" / "main.rs"
RECORD_SOURCE = ROOT / "tools" / "train" / "src" / "record.rs"
BULLET_PATCH = ROOT / "tools" / "train" / "patches" / "bullet-training-only.patch"


def run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(command, check=False, text=True, capture_output=capture)
    if result.returncode:
        if capture:
            sys.stderr.write(result.stderr)
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout.strip() if capture else ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_source_lockfile(path: Path) -> None:
    """Drop lockfile-only references to loaders removed from the sanitized tree."""
    original = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)(?=^\[\[package\]\]$)", original)
    if len(sections) < 2 or not sections[0].startswith("# This file is automatically @generated"):
        raise RuntimeError("Bullet source lockfile has an unexpected generated Cargo.lock format")

    sanitized = [sections[0]]
    removed_packages = {"bullet-utils", "montyformat", "sfbinpack", "viriformat", "bulletformat"}
    seen_removed: set[str] = set()
    for section in sections[1:]:
        parsed = tomllib.loads(section)
        packages = parsed.get("package", [])
        if len(packages) != 1 or not isinstance(packages[0].get("name"), str):
            raise RuntimeError("Bullet source lockfile contains a malformed package stanza")
        name = packages[0]["name"]
        if name in removed_packages:
            seen_removed.add(name)
            continue
        if name == "bullet_lib":
            dependencies = packages[0].get("dependencies", [])
            if any(any(dependency == forbidden or dependency.startswith(forbidden + " ") for forbidden in removed_packages - {"bullet-utils"}) for dependency in dependencies):
                stanza = section
                for forbidden in removed_packages - {"bullet-utils"}:
                    stanza = stanza.replace(f' "{forbidden}",\n', "")
                sanitized.append(stanza)
                continue
            raise RuntimeError("Bullet lockfile bullet_lib stanza did not contain expected loader references")
        sanitized.append(section)

    if seen_removed != removed_packages:
        raise RuntimeError(f"Bullet source lockfile loader package set mismatch: {sorted(seen_removed)}")
    path.write_text("".join(sanitized), encoding="utf-8")
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    names = {package["name"] for package in parsed.get("package", [])}
    if names.intersection(removed_packages):
        raise RuntimeError("sanitized Bullet lockfile still contains a removed loader package")
    bullet_lib = next((package for package in parsed.get("package", []) if package["name"] == "bullet_lib"), None)
    if bullet_lib is None or any(name in dependency for dependency in bullet_lib.get("dependencies", []) for name in removed_packages - {"bullet-utils"}):
        raise RuntimeError("sanitized Bullet lockfile retains a loader dependency")


def assert_no_loader_references(root: Path) -> None:
    forbidden = tuple(name.encode("ascii") for name in FORBIDDEN_LOADER_NAMES)
    for path in root.rglob("*"):
        if path.is_file():
            content = path.read_bytes().lower()
            if any(name in content for name in forbidden):
                raise RuntimeError(f"forbidden chess-format loader reference remains in sanitized Bullet source: {path.relative_to(root)}")


def validate_bullet_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=True)
    if not (root / "Cargo.toml").is_file():
        raise RuntimeError(f"not a Bullet source checkout root: {root}")
    top = Path(run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture=True)).resolve(strict=True)
    if top != root:
        raise RuntimeError(f"Bullet path is not the Git checkout root: {root}")
    revision = run(["git", "-C", str(root), "rev-parse", "HEAD"], capture=True)
    if revision != PINNED_BULLET_COMMIT:
        raise RuntimeError(f"Bullet commit mismatch: expected {PINNED_BULLET_COMMIT}, found {revision}")
    patch_bytes = BULLET_PATCH.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != PINNED_BULLET_PATCH_SHA256:
        raise RuntimeError(f"sanitizer patch hash mismatch: {BULLET_PATCH}")
    return root


def materialize_sanitized_bullet(source: Path, destination: Path) -> tuple[Path, bytes, str]:
    """Export pinned Bullet, apply the patch, and keep only its generic trainer path."""
    destination.mkdir()
    archive = subprocess.Popen(
        ["git", "-C", str(source), "archive", "--format=tar", PINNED_BULLET_COMMIT],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    try:
        extraction = subprocess.run(
            ["tar", "-x", "-f", "-", "-C", str(destination)],
            check=False,
            stdin=archive.stdout,
        )
    finally:
        archive.stdout.close()
    archive_status = archive.wait()
    if archive_status != 0 or extraction.returncode != 0:
        raise RuntimeError(f"could not export pinned Bullet source (git={archive_status}, tar={extraction.returncode})")

    subprocess.run(["git", "apply", "--unidiff-zero", "--whitespace=nowarn"], cwd=destination, input=BULLET_PATCH.read_bytes(), check=True)
    bullet_lib = destination / "crates" / "bullet_lib"
    removed_chess_format_sources = [
        bullet_lib / "src" / "game" / "outputs.rs",
        bullet_lib / "src" / "game" / "inputs" / "adapter.rs",
        bullet_lib / "src" / "game" / "inputs" / "ataxx147.rs",
        bullet_lib / "src" / "game" / "inputs" / "chess768.rs",
        bullet_lib / "src" / "game" / "inputs" / "chess_buckets.rs",
    ]
    for path in removed_chess_format_sources:
        if not path.is_file():
            raise RuntimeError(f"expected Bullet record/feature helper source is missing: {path.relative_to(destination)}")
        path.unlink()

    removed = destination / "crates" / "bullet_lib" / "src" / "value"
    removed_files = [destination / "crates" / "bullet_lib" / "src" / "value.rs"]
    if removed.exists():
        removed_files.extend(sorted(path for path in removed.rglob("*") if path.is_file()))
    for path in removed_files:
        path.unlink(missing_ok=True)
    shutil.rmtree(removed, ignore_errors=True)
    if any(path.exists() for path in removed_files) or removed.exists():
        raise RuntimeError("Bullet value/loader source was not fully removed from sanitized runtime")

    for relative in ("crates/utils", "examples", "docs/2-getting-started.md", "docs/3-data.md"):
        path = destination / relative
        if not path.exists():
            raise RuntimeError(f"expected Bullet loader/example source is missing before sanitization: {relative}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    sanitize_source_lockfile(destination / "Cargo.lock")
    assert_no_loader_references(destination)

    if not (destination / "Cargo.lock").is_file():
        raise RuntimeError("pinned Bullet archive is missing Cargo.lock")
    manifest_lines: list[str] = []
    for path in sorted(destination.rglob("*"), key=lambda item: item.relative_to(destination).as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"symlink is not permitted in sanitized Bullet source: {path.relative_to(destination)}")
        if path.is_file():
            relative = path.relative_to(destination).as_posix()
            manifest_lines.append(f"{file_sha256(path)}\t{relative}\n")
    manifest_bytes = "".join(manifest_lines).encode("utf-8")
    tree_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return destination, manifest_bytes, tree_sha256


def write_runtime_manifest(destination: Path, bullet: Path, backend: str, driver_source: Path) -> Path:
    features = ""
    if backend != "mock":
        features = f", features = [{json.dumps(backend)}]"
    manifest = f'''[package]
name = "kira-m4-trainer-runtime"
version = "0.1.0"
edition = "2024"
publish = false

[dependencies]
bullet-lib = {{ package = "bullet_lib", path = {json.dumps(str(bullet / "crates" / "bullet_lib"))}{features} }}
bullet-compiler = {{ path = {json.dumps(str(bullet / "crates" / "compiler"))} }}
bullet-gpu = {{ path = {json.dumps(str(bullet / "crates" / "gpu"))} }}
bullet-trainer = {{ path = {json.dumps(str(bullet / "crates" / "trainer"))} }}
[[bin]]
name = "kira-m4-trainer"
path = {json.dumps(str(driver_source))}
'''
    path = destination / "Cargo.toml"
    path.write_text(manifest, encoding="utf-8")
    shutil.copy2(bullet / "Cargo.lock", destination / "Cargo.lock")
    return path


TRAINER_VALUE_FLAGS = {
    "--data-dir",
    "--out",
    "--epochs",
    "--patience",
    "--seed",
    "--threads",
    "--validation-batch-size",
}


def parse_training_output(arguments: list[str]) -> tuple[Path, bool]:
    """Mirror Rust option-value consumption and last --out/default semantics."""
    output_dir = Path.cwd() / "m4-training-run"
    index = 0
    validate_only = False
    value_flags = TRAINER_VALUE_FLAGS
    while index < len(arguments):
        argument = arguments[index]
        if argument in value_flags:
            if index + 1 >= len(arguments):
                raise RuntimeError(f"{argument} requires a value")
            value = arguments[index + 1]
            if argument == "--out":
                if value == "" or value in ("--help", "-h") or value.startswith("--"):
                    raise RuntimeError("--out value cannot be empty, help, or another option")
                output_dir = Path(value)
            index += 2
            continue
        if argument.startswith("--out="):
            raise RuntimeError("the Rust trainer requires --out PATH; --out=PATH is unsupported")
        if argument == "--validate-only":
            validate_only = True
        index += 1
    return output_dir.resolve(), validate_only


def trainer_help_requested(arguments: list[str]) -> bool:
    """Match Rust's help handling while respecting flags that consume values."""
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in TRAINER_VALUE_FLAGS:
            index += 2
            continue
        if argument in ("--help", "-h"):
            return True
        index += 1
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bullet-root", type=Path, default=os.environ.get("KIRA_M4_BULLET_ROOT"))
    parser.add_argument("--driver-source", type=Path, default=DRIVER_SOURCE,
                        help="trainer source file to build (defaults to the workspace driver)")
    parser.add_argument("--backend", choices=("mock", "metal", "cuda", "rocm"), default="metal")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="compile the driver but do not execute it")
    mode.add_argument("--test-only", action="store_true", help="run offline unit tests; no optimizer training")
    parser.add_argument("--keep-build-dir", action="store_true", help="retain the isolated temporary Cargo project")
    options, driver_args = parser.parse_known_args()
    if options.bullet_root is None:
        parser.error("--bullet-root or KIRA_M4_BULLET_ROOT is required")
    if driver_args and driver_args[0] == "--":
        driver_args = driver_args[1:]
    if not (options.check_only or options.test_only) and not driver_args:
        parser.error("pass trainer arguments after --; use -- --help for trainer options")
    help_requested = trainer_help_requested(driver_args)
    if help_requested:
        validate_only = False
        output_dir = None
    else:
        parsed_output_dir, validate_only = parse_training_output(driver_args)
        output_dir = None if validate_only else parsed_output_dir

    build_parent: Path | None = None
    try:
        driver_source = options.driver_source.expanduser().resolve(strict=True)
        if not driver_source.is_file():
            raise RuntimeError(f"trainer source is not a file: {driver_source}")
        bullet_source = validate_bullet_root(options.bullet_root)
        build_parent = Path(tempfile.mkdtemp(prefix="kira-m4-driver-", dir="/private/tmp"))
        bullet_root, source_manifest_bytes, source_tree_sha256 = materialize_sanitized_bullet(
            bullet_source, build_parent / "bullet-source"
        )
        source_manifest = build_parent / "bullet-source-manifest.tsv"
        source_manifest.write_bytes(source_manifest_bytes)
        manifest = write_runtime_manifest(build_parent, bullet_root, options.backend, driver_source)
        target = build_parent / "target"
        # Resolve the lockfile for this isolated path manifest, then freeze the
        # resulting runtime graph before any build or test is allowed to run.
        run(["cargo", "tree", "--offline", "--manifest-path", str(manifest)])
        dependency_tree = build_parent / "bullet-runtime-dependency-tree.txt"
        tree_output = run(
            ["cargo", "tree", "--locked", "--offline", "--manifest-path", str(manifest)],
            capture=True,
        )
        if any(name in tree_output for name in ("sfbinpack", "montyformat", "viriformat", "bulletformat")):
            raise RuntimeError("sanitized Bullet dependency tree still contains a forbidden chess-format loader")
        dependency_tree.write_text(tree_output + "\n", encoding="utf-8")
        if options.check_only:
            command = ["cargo", "check", "--locked", "--offline", "--manifest-path", str(manifest), "--target-dir", str(target), "--bin", "kira-m4-trainer"]
        elif options.test_only:
            command = ["cargo", "test", "--locked", "--offline", "--manifest-path", str(manifest), "--target-dir", str(target), "--bin", "kira-m4-trainer"]
        else:
            command = ["cargo", "build", "--release", "--locked", "--offline", "--manifest-path", str(manifest), "--target-dir", str(target), "--bin", "kira-m4-trainer"]
        run(command)
        if not (options.check_only or options.test_only):
            runtime_env = os.environ.copy()
            runtime_env.update(
                {
                    "KIRA_M4_BULLET_ROOT": str(bullet_root),
                    "KIRA_M4_BULLET_COMMIT": PINNED_BULLET_COMMIT,
                    "KIRA_M4_BULLET_LOCK_SHA256": file_sha256(bullet_root / "Cargo.lock"),
                    "KIRA_M4_BULLET_PATCH_SHA256": PINNED_BULLET_PATCH_SHA256,
                    "KIRA_M4_BULLET_SOURCE_MANIFEST": str(source_manifest),
                    "KIRA_M4_BULLET_SOURCE_MANIFEST_SHA256": file_sha256(source_manifest),
                    "KIRA_M4_BULLET_SOURCE_TREE_SHA256": source_tree_sha256,
                    "KIRA_M4_BULLET_DEPENDENCY_TREE": str(dependency_tree),
                    "KIRA_M4_BULLET_DEPENDENCY_TREE_SHA256": file_sha256(dependency_tree),
                    "KIRA_M4_DRIVER_SOURCE_SHA256": file_sha256(driver_source),
                    "KIRA_M4_RECORD_SOURCE_SHA256": file_sha256(RECORD_SOURCE),
                    "KIRA_M4_RUNNER_SHA256": file_sha256(Path(__file__).resolve()),
                    "KIRA_M4_RUNNER_PATH": str(Path(__file__).resolve()),
                    "KIRA_M4_RUNTIME_MANIFEST_SHA256": file_sha256(manifest),
                    "KIRA_M4_RUNTIME_LOCK_SHA256": file_sha256(build_parent / "Cargo.lock"),
                }
            )
            executable = target / "release" / "kira-m4-trainer"
            result = subprocess.run([str(executable), *driver_args], check=False, env=runtime_env)
            if result.returncode:
                raise RuntimeError(f"trainer exited with status {result.returncode}")
            if not validate_only and not help_requested:
                if output_dir is None or not output_dir.is_dir():
                    raise RuntimeError("trainer completed but its --out directory could not be resolved for provenance artifacts")
                shutil.copy2(source_manifest, output_dir / "bullet-source-manifest.tsv")
                shutil.copy2(dependency_tree, output_dir / "bullet-runtime-dependency-tree.txt")
                shutil.copy2(build_parent / "Cargo.lock", output_dir / "bullet-runtime-Cargo.lock")
                shutil.copy2(manifest, output_dir / "bullet-runtime-Cargo.toml")
        print(f"bullet_commit={PINNED_BULLET_COMMIT}")
        print(f"bullet_patch_sha256={PINNED_BULLET_PATCH_SHA256}")
        print(f"bullet_source_tree_sha256={source_tree_sha256}")
        if dependency_tree.is_file():
            print(f"bullet_dependency_tree_sha256={file_sha256(dependency_tree)}")
        if options.keep_build_dir:
            print(f"build_dir={build_parent}")
        else:
            shutil.rmtree(build_parent)
            print("build_dir=removed_after_completion")
        return 0
    except (OSError, RuntimeError) as error:
        if build_parent is not None and not options.keep_build_dir:
            shutil.rmtree(build_parent, ignore_errors=True)
        print(f"kira-m4-driver: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
