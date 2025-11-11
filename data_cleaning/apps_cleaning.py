from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd

NA_STRINGS = ["nan", "NaN", "NULL", "null", "None", "", "Varies with device"]


def _csv_line_numbers(idx: pd.Index) -> List[int]:
    """Convert 0-based DataFrame index to 1-based CSV line numbers.

    Header is line 1, so the first data row is line 2.

    Args:
        idx (pd.Index): Row index from the original DataFrame.

    Returns:
        List[int]: 1-based CSV line numbers.
    """
    return (idx.astype(int) + 2).tolist()


def _clean_installs(s: pd.Series) -> Tuple[pd.Series, pd.Index]:
    """Parse `Installs`: drop commas/plus and cast to nullable Int64.

    Args:
        s (pd.Series): The `Installs` column.

    Returns:
        Tuple[pd.Series, pd.Index]: A tuple of
            (cleaned_series, na_index) where `cleaned_series` is Int64
            and `na_index` are rows that became NA after parsing.
    """
    raw = s.astype("string")
    cleaned = raw.str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    out = pd.to_numeric(cleaned, errors="coerce").astype("Int64")
    na_idx = out[out.isna()].index
    return out, na_idx


def _clean_price(s: pd.Series) -> Tuple[pd.Series, pd.Index]:
    """Normalize `Price`: 'Free'→0, strip '$', cast to float.

    Args:
        s (pd.Series): The `Price` column.

    Returns:
        Tuple[pd.Series, pd.Index]: A tuple of
            (cleaned_series, na_index) where `cleaned_series` is float
            and `na_index` are rows that became NA after parsing.
    """
    raw = s.astype("string").str.strip()
    tmp = raw.mask(raw.str.lower() == "free", "0").str.replace("$", "", regex=False)
    out = pd.to_numeric(tmp, errors="coerce")
    na_idx = out[out.isna()].index
    return out, na_idx


def _coerce_rating_0_5(s: pd.Series) -> Tuple[pd.Series, pd.Index]:
    """Clip `Rating` to [0, 5]; out-of-range values become NA.

    Args:
        s (pd.Series): The `Rating` column.

    Returns:
        Tuple[pd.Series, pd.Index]: A tuple of
            (cleaned_series, bad_index) where `bad_index` are rows set to NA.
    """
    out = pd.to_numeric(s, errors="coerce")
    bad = out[(out < 0) | (out > 5)].index
    out.loc[bad] = pd.NA
    return out, bad


def _parse_last_updated(s: pd.Series) -> pd.Series:
    """Parse `Last Updated` into pandas datetime (UTC-naive).

    Unparseable values become NaT.

    Args:
        s (pd.Series): The `Last Updated` column.

    Returns:
        pd.Series: Datetime64[ns] with NaT for invalid rows.
    """
    return pd.to_datetime(s, errors="coerce", utc=False)


def head_with_csv_lines(in_csv: Union[str, Path], n: int = 5) -> pd.DataFrame:
    """Preview the first n data rows with CSV line numbers.

    Header counts as line 1.

    Args:
        in_csv (str | Path): Path to the CSV file.
        n (int): Number of data rows to return. Defaults to 5.

    Returns:
        pd.DataFrame: Preview with an extra `csv_line` column.
    """
    df = pd.read_csv(in_csv)
    out = df.head(n).copy()
    out.insert(0, "csv_line", range(2, 2 + len(out)))
    return out


def clean_googleplay_apps(
    in_csv: Union[str, Path],
    out_csv: Optional[Union[str, Path]] = None,
    keep_latest_per_app: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Clean the Google Play Apps metadata and return (df_clean, report).

    Pipeline:
        1) Drop exact duplicate rows.
        2) For duplicate `App`, keep the latest by `Last Updated` (tie-break by `Reviews`).
        3) `Rating` → clip to [0, 5].
        4) `Installs` → remove commas/plus, cast to Int64.
        5) `Price` → 'Free'→0, strip '$', cast to float.
        6) `Size` → mark NA if blank/special token (no unit conversion here).
        7) Trim whitespace for string columns.
        8) Count NA/anomalies and record original CSV line numbers.

    Args:
        in_csv (str | Path): Path to raw apps CSV.
        out_csv (str | Path | None): If set, write the cleaned CSV.
        keep_latest_per_app (bool): Whether to keep the latest record per `App`.
            Defaults to True.

    Returns:
        Tuple[pd.DataFrame, Dict[str, object]]
    """

    df0 = pd.read_csv(in_csv, dtype=str, na_values=NA_STRINGS, keep_default_na=True)
    df0 = df0.reset_index(drop=True)

    report: Dict[str, object] = {}
    report["rows_in"] = len(df0)

    # (1) Drop exact duplicate rows
    dup_rows_mask = df0.duplicated(keep="first")
    dup_rows_idx = df0.index[dup_rows_mask]
    df1 = df0.loc[~dup_rows_mask].copy()
    report["dup_rows_dropped"] = int(dup_rows_mask.sum())
    report["dup_rows_lines"] = _csv_line_numbers(dup_rows_idx)


    for c in df1.columns:
        if pd.api.types.is_string_dtype(df1[c]):
            df1[c] = df1[c].astype("string").str.strip()


    if "Last Updated" in df1.columns:
        df1["_LastUpdated_dt"] = _parse_last_updated(df1["Last Updated"])
    else:
        df1["_LastUpdated_dt"] = pd.NaT

    if "Reviews" in df1.columns:
        df1["_Reviews_num"] = pd.to_numeric(df1["Reviews"], errors="coerce")
    else:
        df1["_Reviews_num"] = pd.NA

    # (2) Drop duplicate Apps (group by 'App')
    if "App" in df1.columns:
        dup_app_mask_by_first = df1.duplicated(subset=["App"], keep="first")
        dup_app_idx_all = df1.index[dup_app_mask_by_first]

        if keep_latest_per_app:
            df1["_orig_pos"] = np.arange(len(df1))
            sort_cols = ["_LastUpdated_dt", "_Reviews_num", "_orig_pos"]
            df_sorted = df1.sort_values(
                sort_cols,
                ascending=[False, False, True],
                kind="mergesort",
            )
            kept_idx = (
                df_sorted.drop_duplicates(subset=["App"], keep="first")
                .sort_values("_orig_pos")
                .index
            )
            drop_idx = df1.index.difference(kept_idx)
            df1 = df1.loc[kept_idx].copy().sort_values("_orig_pos").drop(columns=["_orig_pos"])
            report["dup_apps_dropped"] = len(drop_idx)
            report["dup_apps_lines"] = _csv_line_numbers(drop_idx)
        else:
            report["dup_apps_dropped"] = int(dup_app_mask_by_first.sum())
            report["dup_apps_lines"] = _csv_line_numbers(dup_app_idx_all)
            df1 = df1.loc[~dup_app_mask_by_first].copy()
    else:
        report["dup_apps_dropped"] = 0
        report["dup_apps_lines"] = []

    # (3) Rating
    if "Rating" in df1.columns:
        df1["Rating"], bad_rating_idx = _coerce_rating_0_5(df1["Rating"])
        report["rating_out_of_range"] = len(bad_rating_idx)
        report["rating_bad_lines"] = _csv_line_numbers(bad_rating_idx)
    else:
        report["rating_out_of_range"] = 0
        report["rating_bad_lines"] = []

    # (4) Installs
    if "Installs" in df1.columns:
        df1["Installs"], na_inst_idx = _clean_installs(df1["Installs"])
        report["installs_na"] = len(na_inst_idx)
        report["installs_na_lines"] = _csv_line_numbers(na_inst_idx)
    else:
        report["installs_na"] = 0
        report["installs_na_lines"] = []

    # (5) Price
    if "Price" in df1.columns:
        df1["Price"], na_price_idx = _clean_price(df1["Price"])
        report["price_na"] = len(na_price_idx)
        report["price_na_lines"] = _csv_line_numbers(na_price_idx)
    else:
        report["price_na"] = 0
        report["price_na_lines"] = []

    # (6) Size NA count (unit conversion is handled in apps_basic_stats)
    if "Size" in df1.columns:
        size = df1["Size"].astype("string")
        size_na_mask = size.isna() | (size.str.strip() == "")
        na_size_idx = df1.index[size_na_mask]
        report["size_na"] = int(size_na_mask.sum())
        report["size_na_lines"] = _csv_line_numbers(na_size_idx)
    else:
        report["size_na"] = 0
        report["size_na_lines"] = []

    # (7) Simple NA counts for Type / Content Rating
    for col, key in [("Type", "type_na"), ("Content Rating", "content_rating_na")]:
        if col in df1.columns:
            na_mask = df1[col].isna() | (df1[col].astype("string").str.strip() == "")
            report[key] = int(na_mask.sum())
            report[f"{key}_lines"] = _csv_line_numbers(df1.index[na_mask])
        else:
            report[key] = 0
            report[f"{key}_lines"] = []


    for c in ["_LastUpdated_dt", "_Reviews_num"]:
        if c in df1.columns:
            df1 = df1.drop(columns=[c])


    report["rows_out"] = len(df1)

    # Save
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df1.to_csv(out_csv, index=False)

    return df1, report


def apps_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic numeric stats for apps.

    Columns considered: `Rating`, `Reviews`, `Installs`, `Price`, and `Size_MB`
    (derived from `Size`). Output columns: `column, count_non_null, count_na,
    min, max, mean, std`.

    Args:
        df (pd.DataFrame): Cleaned apps DataFrame.

    Returns:
        pd.DataFrame: Summary statistics table.
    """
    def _size_to_mb(s: pd.Series) -> pd.Series:
        """Convert `Size` strings (e.g., '14M', '512k') to MB as Float64.

            - 'M' → MB
            - 'K' → KB (divided by 1024)
            - blank/special tokens → NA

        Args:
            s (pd.Series): The `Size` column.

        Returns:
            pd.Series: Float64 MB values with NA where unparseable.
        """
        ser = s.astype("string")

        def parse_one(x: str):
            if x is None or pd.isna(x):
                return pd.NA
            t = x.strip().replace(" ", "").upper()
            if t == "" or t in {"VARIESWITHDEVICE", "VARIES", "NAN"}:
                return pd.NA
            try:
                if t.endswith("M"):
                    return float(t[:-1])
                if t.endswith("K"):
                    return float(t[:-1]) / 1024.0
                return float(t)  # plain number fallback
            except Exception:
                return pd.NA

        out = ser.map(parse_one)
        return out.astype("Float64")

    cols: Dict[str, pd.Series] = {}
    if "Rating" in df.columns:
        cols["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    if "Reviews" in df.columns:
        cols["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")
    if "Installs" in df.columns:
        cols["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")
    if "Price" in df.columns:
        cols["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    if "Size" in df.columns:
        cols["Size_MB"] = _size_to_mb(df["Size"])

    rows = []
    for name, s in cols.items():
        cnt_non_null = int(s.notna().sum())
        cnt_na = int(s.isna().sum())
        vmin = float(np.nanmin(s)) if cnt_non_null else np.nan
        vmax = float(np.nanmax(s)) if cnt_non_null else np.nan
        vmean = float(np.nanmean(s)) if cnt_non_null else np.nan
        vstd = float(np.nanstd(s, ddof=1)) if cnt_non_null > 1 else np.nan
        rows.append({
            "column": name,
            "count_non_null": cnt_non_null,
            "count_na": cnt_na,
            "min": vmin,
            "max": vmax,
            "mean": vmean,
            "std": vstd,
        })

    out = pd.DataFrame(rows, columns=[
        "column", "count_non_null", "count_na", "min", "max", "mean", "std"
    ])

    order = ["Rating", "Reviews", "Installs", "Price", "Size_MB"]
    out["__ord"] = out["column"].apply(lambda c: order.index(c) if c in order else 999)
    out = out.sort_values(["__ord", "column"]).drop(columns="__ord").reset_index(drop=True)
    return out
