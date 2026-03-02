from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.io.read_csv_chunks import read_csv_in_chunks, get_read_params


def step_00_profile_raw(cfg: Dict[str, Any]) -> Dict[str, Any]:
    
    
    params = get_read_params(cfg)
    file_path = params["file_path"]

    paths = cfg["paths"]
    reports_dir = Path(paths["reports"])
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = params["chunk_size"]
    encoding = params["encoding"]
    sep = params["sep"]
    decimal = params["decimal"]

    total_rows = 0
    col_names = None

    missing_sum = None
    total_rows_per_col = None

    for chunk in read_csv_in_chunks(
        file_path=file_path,
        chunk_size=chunk_size,
        encoding=encoding,
        sep=sep,
        decimal=decimal,
    ):
        if col_names is None:
            col_names = list(chunk.columns)
            missing_sum = pd.Series(0, index=col_names, dtype="int64")
            total_rows_per_col = pd.Series(0, index=col_names, dtype="int64")

        total_rows += len(chunk)
        missing_sum = (missing_sum + chunk.isna().sum()).astype("int64")
        total_rows_per_col = (total_rows_per_col + len(chunk)).astype("int64")

    if col_names is None:
        raise RuntimeError(f"No data read from: {file_path}")

    missing_ratio = (missing_sum / total_rows_per_col).sort_values(ascending=False)

    summary = pd.DataFrame(
        {
            "file_path": [file_path],
            "n_rows": [total_rows],
            "n_cols": [len(col_names)],
            "chunk_size": [chunk_size],
        }
    )

    summary_path = tables_dir / "raw_profile_summary.csv"
    missing_path = tables_dir / "raw_missing_ratio.csv"

    summary.to_csv(summary_path, index=False)
    missing_ratio.rename("missing_ratio").to_csv(missing_path)

    return {
        "n_rows": total_rows,
        "n_cols": len(col_names),
        "summary_path": str(summary_path),
        "missing_path": str(missing_path),
    }