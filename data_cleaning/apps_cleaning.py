from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd

NA_STRINGS = ["nan", "NaN", "NULL", "null", "None", ""]

def _csv_line_numbers(idx: pd.Index) -> List[int]:
    """Convert 0-based DataFrame index to 1-based CSV line numbers."""
    return (idx.astype(int) + 2).tolist()


def _clean_installs(s: pd.Series) -> Tuple[pd.Series, pd.Index]:
    """Parse `Installs`: drop commas/plus and cast to nullable Int64."""
    raw = s.astype("string")
    cleaned = raw.str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    out = pd.to_numeric(cleaned, errors="coerce").astype("Int64")
    na_idx = out[out.isna()].index
    return out, na_idx


def _clean_price(s: pd.Series) -> Tuple[pd.Series, pd.Index]:
    """Normalize `Price`: 'Free'→0, strip '$', cast to float."""
    raw = s.astype("string").str.strip()
    tmp = raw.mask(raw.str.lower() == "free", "0").str.replace("$", "", regex=False)
    out = pd.to_numeric(tmp, errors="coerce")
    na_idx = out[out.isna()].index
    return out, na_idx


def _coerce_rating_0_5(s: pd.Series) -> Tuple[pd.Series, pd.Index]:
    """Clip `Rating` to [0, 5]; out-of-range values become NA. """
    out = pd.to_numeric(s, errors="coerce")
    bad = out[(out < 0) | (out > 5)].index
    out.loc[bad] = pd.NA
    return out, bad

def _bucket_versions(x: pd.Series) ->pd.Series:
  """bucketing each version to their closest int value and filling the missing with the median values"""
  x_clean = x.astype("string").str.strip().str[0]
  x_bucketed = pd.to_numeric(x_clean,errors='coerce')
  x_bucketed.fillna(x_bucketed.median(numeric_only=True), inplace=True)
  na_idx = x_bucketed[x_bucketed.isna()].index
  return x_bucketed,na_idx


def _parse_last_updated(s: pd.Series) -> pd.Series:
    """Parse `Last Updated` into pandas datetime (UTC-naive)."""
    return pd.to_datetime(s, errors="coerce", utc=False)


def clean_googleplay_apps(
    in_csv: Union[str, Path],
    out_csv: Optional[Union[str, Path]] = None,
    keep_latest_per_app: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Clean the Google Play Apps"""

    df0 = pd.read_csv(in_csv,na_values=NA_STRINGS, keep_default_na=True)
    df0 = df0.reset_index(drop=True)
    

    report: Dict[str, object] = {}
    report["rows_in"] = len(df0)

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

    if "Rating" in df1.columns:
        df1["Rating"], bad_rating_idx = _coerce_rating_0_5(df1["Rating"])
        report["rating_out_of_range"] = len(bad_rating_idx)
        report["rating_bad_lines"] = _csv_line_numbers(bad_rating_idx)
    else:
        report["rating_out_of_range"] = 0
        report["rating_bad_lines"] = []

    if "Installs" in df1.columns:
        df1["Installs"], na_inst_idx = _clean_installs(df1["Installs"])
        report["installs_na"] = len(na_inst_idx)
        report["installs_na_lines"] = _csv_line_numbers(na_inst_idx)
    else:
        report["installs_na"] = 0
        report["installs_na_lines"] = []

    if "Price" in df1.columns:
        df1["Price"], na_price_idx = _clean_price(df1["Price"])
        report["price_na"] = len(na_price_idx)
        report["price_na_lines"] = _csv_line_numbers(na_price_idx)
    else:
        report["price_na"] = 0
        report["price_na_lines"] = []

    if "Size" in df1.columns:
        df1["Size"] = _size_to_mb(df1["Size"])
        na_size_idx=df1[df1["Size"].isna()].index
        report["size_na"]=len(na_size_idx)
        report["size_na_lines"] = _csv_line_numbers(na_size_idx)
    else:
        report["size_na"] = 0
        report["size_na_lines"] = []


    if "Android Ver" in df1.columns:
        df1["Android Ver"], na_and_vr_idx = _bucket_versions(df1["Android Ver"])
        report["android_ver_na"] = len(na_and_vr_idx)
        report["android_ver_na_lines"] = _csv_line_numbers(na_and_vr_idx)
    else:
        report["android_ver_na"] = 0
        report["android_ver_na_lines"] = [] 
  
    df1["Popularity Score"] = df1["Rating"] * np.log1p(df1["Installs"])
    df1["Popularity Score"]  =  pd.qcut(df1["Popularity Score"], 5, labels=[1,2,3,4,5])

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

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df1.to_csv(out_csv, index=False)

    return df1, report

def _size_to_mb(s: pd.Series) -> pd.Series:
        """Convert `Size` strings (e.g., '14M', '512k') to MB as Float64."""

        ser = s.astype(str)
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
                return float(t) 
            except Exception:
                return pd.NA

        out = ser.map(parse_one)
        median_val = out.median()
        out = out.fillna(median_val)
        return out.astype("Float64")

def apps_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic numeric stats for apps."""
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
        cols["Size_MB"] =pd.to_numeric(df["Size"], errors="coerce")
    if "Android Ver" in df.columns:
        cols["Android Ver"]=df["Android Ver"]
     

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

