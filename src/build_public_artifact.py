from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config/public_artifact_manifest.json"
MAX_FILE_BYTES = 20 * 1024 * 1024
FORBIDDEN_ROOTS = {
    ".git",
    ".pytest_cache",
    "archive",
    "build",
    "data",
    "literature",
    "logs",
    "models",
    "tmp",
}
FORBIDDEN_NAME_FRAGMENTS = {
    "automatic_key",
    "blind_review",
    "candidate_set_audit",
    "context_sample",
    "debug",
    "evaluation_item_audit",
    "evaluation_items",
    "exposure_items",
    "matched_pairs",
    "mvp_per_example",
    "nonce_per_example",
    "smoke",
}
FORBIDDEN_SUFFIXES = {".doc", ".docx", ".log", ".safetensors"}
TEXT_SUFFIXES = {
    ".bib",
    ".bst",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sty",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_PATH_MARKERS = ("/" + "Users" + "/", "C:" + "\\" + "Users" + "\\")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise RuntimeError(f"Manifest path must be project-relative: {value!r}")
    return Path(*pure.parts)


def validate_public_path(relative: Path, source: Path) -> None:
    if relative.parts[0] in FORBIDDEN_ROOTS:
        raise RuntimeError(f"Forbidden artifact root: {relative}")
    lowered = relative.name.lower()
    if source.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"Forbidden artifact suffix: {relative}")
    if any(fragment in lowered for fragment in FORBIDDEN_NAME_FRAGMENTS):
        raise RuntimeError(f"Forbidden artifact filename: {relative}")
    if source.is_symlink():
        raise RuntimeError(f"Symlinks are not allowed in the artifact: {relative}")
    if not source.is_file():
        raise RuntimeError(f"Allowlisted artifact is not a regular file: {relative}")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise RuntimeError(f"Artifact exceeds {MAX_FILE_BYTES} bytes: {relative}")
    try:
        source.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"Artifact escapes project root: {relative}") from error
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8")
        marker = next((item for item in LOCAL_PATH_MARKERS if item in text), None)
        if marker:
            raise RuntimeError(f"Machine-specific absolute path found in {relative}")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"artifact_name", "required_files", "required_globs", "e7_files"}
    missing = required - set(manifest)
    if missing:
        raise RuntimeError(f"Manifest fields missing: {sorted(missing)}")
    if not isinstance(manifest["artifact_name"], str) or not manifest["artifact_name"].strip():
        raise RuntimeError("artifact_name must be a non-empty string")
    return manifest


def resolve_manifest(manifest: dict[str, Any], require_e7: bool) -> list[Path]:
    relative_paths = {normalize_relative(value) for value in manifest["required_files"]}
    for pattern in manifest["required_globs"]:
        matched = [path for path in ROOT.glob(pattern) if path.is_file()]
        if not matched:
            raise RuntimeError(f"Required manifest glob matched no files: {pattern}")
        relative_paths.update(path.relative_to(ROOT) for path in matched)
    if require_e7:
        relative_paths.update(normalize_relative(value) for value in manifest["e7_files"])

    resolved: list[Path] = []
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        source = ROOT / relative
        if not source.exists():
            raise RuntimeError(f"Allowlisted artifact is missing: {relative}")
        validate_public_path(relative, source)
        resolved.append(relative)
    return resolved


def validate_e7_completion() -> None:
    progress_path = ROOT / "results/ACL_confirmatory_aux_v2/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("completed_models") != 25 or progress.get("failures"):
        raise RuntimeError("E7 is not complete: expected 25 checkpoints and no failures")
    readiness_path = ROOT / "results/release_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if readiness.get("status") != "ready_for_selected_gate" or readiness.get("required_failures"):
        raise RuntimeError("The latest release-readiness report does not pass the selected E7 gate")


def build_report(files: list[Path], require_e7: bool) -> dict[str, Any]:
    return {
        "artifact_policy": "code_and_aggregate_results_only",
        "e7_gate_required": require_e7,
        "file_count": len(files),
        "total_source_bytes": sum((ROOT / relative).stat().st_size for relative in files),
        "excluded_roots": sorted(FORBIDDEN_ROOTS),
        "excluded_sensitive_classes": sorted(FORBIDDEN_NAME_FRAGMENTS),
        "source_license_status": json.loads(
            (ROOT / "docs/SOURCE_LICENSE_VERIFICATION.json").read_text(encoding="utf-8")
        ).get("status"),
        "submission_ready": json.loads(
            (ROOT / "results/release_readiness.json").read_text(encoding="utf-8")
        ).get("submission_ready"),
    }


def write_zip_member(archive: zipfile.ZipFile, source: Path, arcname: str, executable: bool) -> None:
    info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, source.read_bytes())


def build_artifact(manifest: dict[str, Any], files: list[Path], force: bool) -> tuple[Path, Path]:
    output_root = ROOT / "output/public_artifact"
    package_dir = output_root / manifest["artifact_name"]
    archive_path = ROOT / "output" / f"{manifest['artifact_name']}.zip"
    if package_dir.exists() or archive_path.exists():
        if not force:
            raise RuntimeError("Artifact output already exists; pass --force to replace generated outputs")
        if package_dir.exists():
            shutil.rmtree(package_dir)
        if archive_path.exists():
            archive_path.unlink()
    package_dir.mkdir(parents=True)

    checksum_lines: list[str] = []
    for relative in files:
        source = ROOT / relative
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksum_lines.append(f"{sha256(destination)}  {relative.as_posix()}")
    checksum_path = package_dir / "MANIFEST.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    report_path = package_dir / "ARTIFACT_BUILD.json"
    report_path.write_text(
        json.dumps(build_report(files, require_e7=True), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive_path, "w") as archive:
        members = files + [Path("MANIFEST.sha256"), Path("ARTIFACT_BUILD.json")]
        for relative in members:
            source = package_dir / relative
            executable = relative.suffix == ".sh"
            arcname = f"{manifest['artifact_name']}/{relative.as_posix()}"
            write_zip_member(archive, source, arcname, executable)
    return package_dir, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the allowlisted, transcript-free ACL public artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-e7", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.check_only and not args.require_e7:
        raise SystemExit("Refusing to build a partial public artifact; use --require-e7 or --check-only")

    manifest = load_manifest(args.manifest)
    if args.require_e7:
        validate_e7_completion()
    files = resolve_manifest(manifest, require_e7=args.require_e7)
    report = build_report(files, require_e7=args.require_e7)
    if args.check_only:
        print(json.dumps({"status": "check_passed", **report}, ensure_ascii=False, indent=2))
        return
    package_dir, archive_path = build_artifact(manifest, files, force=args.force)
    print(
        json.dumps(
            {"status": "built", **report, "directory": str(package_dir), "archive": str(archive_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
