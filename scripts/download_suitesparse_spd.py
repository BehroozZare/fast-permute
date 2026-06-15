#!/usr/bin/env python3
"""Download real square matrices from the SuiteSparse Matrix Collection.

The script uses SuiteSparse's public ``ssstats.csv`` metadata. By default it
downloads a capped, medium benchmark set, extracts the Matrix Market ``.mtx``
files, and removes archives after successful extraction.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STATS_URL = "https://sparse.tamu.edu/files/ssstats.csv"
DEFAULT_MM_BASE_URL = "https://sparse.tamu.edu/MM"
CSV_COLUMNS = [
    "group",
    "name",
    "nrows",
    "ncols",
    "nnz",
    "is_real",
    "is_binary",
    "is_nd",
    "posdef",
    "pattern_symmetry",
    "numerical_symmetry",
    "kind",
    "nentries",
]


@dataclass(frozen=True)
class MatrixInfo:
    group: str
    name: str
    nrows: int
    ncols: int
    nnz: int
    is_real: bool
    is_binary: bool
    is_nd: bool
    posdef: int
    pattern_symmetry: float
    numerical_symmetry: float
    kind: str
    nentries: int

    @property
    def slug(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def estimated_bytes(self) -> int:
        entries = max(self.nentries, self.nnz, 1)
        return entries * 32 + 2048


@dataclass(frozen=True)
class DownloadResult:
    matrix: MatrixInfo
    status: str
    path: str
    archive: str
    message: str = ""


def positive_int_or_none(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def positive_float_or_none(value: str) -> float | None:
    parsed = float(value)
    return None if parsed <= 0.0 else parsed


def read_text_from_url_or_file(location: str, timeout: int) -> str:
    parsed = urllib.parse.urlparse(location)
    if parsed.scheme in ("http", "https"):
        request = urllib.request.Request(
            location, headers={"User-Agent": "homa-spd-downloader/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")

    with open(location, "r", encoding="utf-8") as f:
        return f.read()


def parse_bool_int(value: str) -> bool:
    return int(value) != 0


def parse_stats_csv(text: str) -> list[MatrixInfo]:
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("ssstats.csv is too short")

    matrices: list[MatrixInfo] = []
    reader = csv.reader(lines[2:])
    for line_number, row in enumerate(reader, start=3):
        if not row:
            continue
        if len(row) < len(CSV_COLUMNS):
            raise ValueError(f"Malformed sstats row {line_number}: {row}")

        if len(row) > len(CSV_COLUMNS):
            row = row[:11] + [",".join(row[11:-1]), row[-1]]

        matrices.append(
            MatrixInfo(
                group=row[0],
                name=row[1],
                nrows=int(row[2]),
                ncols=int(row[3]),
                nnz=int(row[4]),
                is_real=parse_bool_int(row[5]),
                is_binary=parse_bool_int(row[6]),
                is_nd=parse_bool_int(row[7]),
                posdef=int(row[8]),
                pattern_symmetry=float(row[9]),
                numerical_symmetry=float(row[10]),
                kind=row[11],
                nentries=int(row[12]),
            )
        )
    return matrices


def filter_square_matrices(
    matrices: list[MatrixInfo],
    max_files: int | None,
    min_rows: int | None,
    max_rows: int | None,
    max_nnz: int | None,
    max_total_gb: float | None,
    include_binary: bool,
) -> list[MatrixInfo]:
    candidates = [
        matrix
        for matrix in matrices
        if matrix.is_real
        and matrix.nrows == matrix.ncols
        and (include_binary or not matrix.is_binary)
        and (min_rows is None or matrix.nrows >= min_rows)
        and (max_rows is None or matrix.nrows <= max_rows)
        and (max_nnz is None or matrix.nnz <= max_nnz)
    ]
    candidates.sort(key=lambda matrix: (matrix.nnz, matrix.nrows, matrix.slug))

    selected: list[MatrixInfo] = []
    estimated_total = 0
    max_total_bytes = None
    if max_total_gb is not None:
        max_total_bytes = int(max_total_gb * 1024 * 1024 * 1024)

    for matrix in candidates:
        if max_files is not None and len(selected) >= max_files:
            break
        next_total = estimated_total + matrix.estimated_bytes
        if max_total_bytes is not None and selected and next_total > max_total_bytes:
            break
        selected.append(matrix)
        estimated_total = next_total

    return selected


def matrix_url(base_url: str, matrix: MatrixInfo) -> str:
    group = urllib.parse.quote(matrix.group)
    name = urllib.parse.quote(matrix.name)
    return f"{base_url.rstrip('/')}/{group}/{name}.tar.gz"


def safe_member_name(member: tarfile.TarInfo) -> str:
    name = member.name.replace("\\", "/")
    parts = [part for part in name.split("/") if part and part not in (".", "..")]
    return "/".join(parts)


def extract_matrix_market_file(archive_path: Path, target_path: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile() and safe_member_name(member).lower().endswith(".mtx")
        ]
        if not members:
            raise RuntimeError(f"No .mtx file found in {archive_path}")

        member = min(members, key=lambda item: len(safe_member_name(item)))
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not extract {member.name} from {archive_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb", delete=False, dir=target_path.parent, prefix=target_path.name + "."
        ) as tmp:
            shutil.copyfileobj(source, tmp)
            tmp_path = Path(tmp.name)

    os.replace(tmp_path, target_path)


def download_archive(url: str, archive_path: Path, timeout: int, force: bool) -> None:
    if archive_path.exists() and not force:
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "homa-spd-downloader/1.0"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(part_path, "wb") as f:
                shutil.copyfileobj(response, f)
        os.replace(part_path, archive_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def download_one(
    matrix: MatrixInfo,
    out_dir: Path,
    base_url: str,
    keep_archives: bool,
    force: bool,
    timeout: int,
) -> DownloadResult:
    target_path = out_dir / matrix.group / f"{matrix.name}.mtx"
    archive_path = out_dir / "_archives" / matrix.group / f"{matrix.name}.tar.gz"

    if target_path.exists() and not force:
        return DownloadResult(matrix, "skipped", str(target_path), str(archive_path))

    url = matrix_url(base_url, matrix)
    try:
        download_archive(url, archive_path, timeout=timeout, force=force)
        extract_matrix_market_file(archive_path, target_path)
        if not keep_archives:
            archive_path.unlink(missing_ok=True)
        return DownloadResult(matrix, "downloaded", str(target_path), str(archive_path))
    except (OSError, tarfile.TarError, urllib.error.URLError, RuntimeError) as exc:
        return DownloadResult(matrix, "failed", str(target_path), str(archive_path), str(exc))


def write_manifest(path: Path, matrices: list[MatrixInfo], base_url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS + ["estimated_mb", "url"])
        for matrix in matrices:
            writer.writerow(
                [
                    matrix.group,
                    matrix.name,
                    matrix.nrows,
                    matrix.ncols,
                    matrix.nnz,
                    int(matrix.is_real),
                    int(matrix.is_binary),
                    int(matrix.is_nd),
                    matrix.posdef,
                    matrix.pattern_symmetry,
                    matrix.numerical_symmetry,
                    matrix.kind,
                    matrix.nentries,
                    f"{matrix.estimated_bytes / (1024 * 1024):.3f}",
                    matrix_url(base_url, matrix),
                ]
            )


def write_results(path: Path, results: list[DownloadResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "name",
                "nrows",
                "ncols",
                "nnz",
                "status",
                "path",
                "archive",
                "message",
            ]
        )
        for result in results:
            matrix = result.matrix
            writer.writerow(
                [
                    matrix.group,
                    matrix.name,
                    matrix.nrows,
                    matrix.ncols,
                    matrix.nnz,
                    result.status,
                    result.path,
                    result.archive,
                    result.message,
                ]
            )


def print_selection(matrices: list[MatrixInfo]) -> None:
    for matrix in matrices:
        print(
            f"{matrix.slug:45s} rows={matrix.nrows:9d} "
            f"nnz={matrix.nnz:10d} kind={matrix.kind}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real square Matrix Market matrices from SuiteSparse."
    )
    parser.add_argument("--out", default="data/suitesparse_spd", help="Output directory")
    parser.add_argument(
        "--stats-url",
        default=DEFAULT_STATS_URL,
        help="SuiteSparse sstats.csv URL or local file path",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_MM_BASE_URL,
        help="Base URL for Matrix Market archives",
    )
    parser.add_argument(
        "--max-files",
        type=positive_int_or_none,
        default=50,
        help="Maximum number of matrices to select; 0 disables this cap",
    )
    parser.add_argument(
        "--min-rows",
        type=positive_int_or_none,
        default=None,
        help="Minimum row count; 0 disables this floor",
    )
    parser.add_argument(
        "--max-rows",
        type=positive_int_or_none,
        default=1_000_000,
        help="Maximum row count; 0 disables this cap",
    )
    parser.add_argument(
        "--max-nnz",
        type=positive_int_or_none,
        default=20_000_000,
        help="Maximum nonzeros; 0 disables this cap",
    )
    parser.add_argument(
        "--max-total-gb",
        type=positive_float_or_none,
        default=25.0,
        help="Approximate extracted Matrix Market size cap; 0 disables this cap",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--timeout", type=int, default=120, help="Network timeout in seconds")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only write the manifest and print selected matrices",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tar.gz archives after extraction",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract matrices even if .mtx files already exist",
    )
    parser.add_argument(
        "--include-binary",
        action="store_true",
        help="Include SuiteSparse binary/pattern square matrices",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        print("--workers must be positive", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    try:
        stats_text = read_text_from_url_or_file(args.stats_url, timeout=args.timeout)
        matrices = parse_stats_csv(stats_text)
    except Exception as exc:
        print(f"Failed to read SuiteSparse stats: {exc}", file=sys.stderr)
        return 1

    selected = filter_square_matrices(
        matrices,
        max_files=args.max_files,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        max_nnz=args.max_nnz,
        max_total_gb=args.max_total_gb,
        include_binary=args.include_binary,
    )

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, selected, args.base_url)
    estimated_gb = sum(matrix.estimated_bytes for matrix in selected) / (1024**3)
    print(
        f"Selected {len(selected)} real square matrices "
        f"(estimated extracted size {estimated_gb:.2f} GiB)"
    )
    print(f"Wrote manifest: {manifest_path}")

    if args.list_only:
        print_selection(selected)
        return 0

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                matrix,
                out_dir,
                args.base_url,
                args.keep_archives,
                args.force,
                args.timeout,
            )
            for matrix in selected
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            label = result.matrix.slug
            if result.status == "failed":
                print(f"[failed]     {label}: {result.message}", file=sys.stderr)
            else:
                print(f"[{result.status:10s}] {label}")

    results.sort(key=lambda result: (result.matrix.nnz, result.matrix.nrows, result.matrix.slug))
    results_path = out_dir / "downloaded.csv"
    write_results(results_path, results)

    failed = sum(1 for result in results if result.status == "failed")
    downloaded = sum(1 for result in results if result.status == "downloaded")
    skipped = sum(1 for result in results if result.status == "skipped")
    print(f"Wrote results: {results_path}")
    print(f"Downloaded: {downloaded}, skipped: {skipped}, failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
