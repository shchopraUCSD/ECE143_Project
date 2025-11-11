# reviews_cleaning.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# Treat these string tokens as missing/NA on read.
NA_STRINGS = ["nan", "NaN", "NULL", "null", "None", ""]

# Canonical labels
_CANON = {"Positive", "Negative", "Neutral"}

# Alias dictionary (lower-cased keys) -> Canonical form
_SENTIMENT_ALIASES = {
    # positive
    "positive": "Positive", "pos": "Positive", "p": "Positive",
    # negative
    "negative": "Negative", "neg": "Negative", "n": "Negative",
    # neutral
    "neutral": "Neutral", "neu": "Neutral", "neut": "Neutral",
}


def normalize_sentiment_labels(df: pd.DataFrame,
                               col: str = "Sentiment") -> tuple[pd.DataFrame, int]:
    """Normalize sentiment labels to {Positive, Negative, Neutral}.

    Mapping is case-insensitive. Unknown/blank values become NA.

    Args:
      df: Input DataFrame.
      col: Sentiment column name.

    Returns:
      (new_df, changed_count)
    """
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")

    ser = df[col].astype("string")

    def _map_one(x: Optional[str]) -> Optional[str]:
        if x is None or pd.isna(x):
            return pd.NA
        s = str(x).strip()
        if s == "":
            return pd.NA
        canon = _SENTIMENT_ALIASES.get(s.lower())
        return canon if canon in _CANON else pd.NA

    before = ser.copy()
    out = df.copy()
    out[col] = ser.map(_map_one)
    changed = int((before != out[col]).sum(skipna=False))
    return out, changed


def drop_all_nan_reviews(in_csv: str | Path) -> tuple[pd.DataFrame, int]:
    """Drop rows where review and all sentiment fields are missing.

    A row is dropped if Translated_Review, Sentiment, Sentiment_Polarity,
    and Sentiment_Subjectivity are all NA.

    Args:
      in_csv: Path to the reviews CSV.

    Returns:
      (df_clean, n_dropped_all4)
    """
    df = pd.read_csv(in_csv, na_values=NA_STRINGS, keep_default_na=True)
    cols = ["Translated_Review", "Sentiment", "Sentiment_Polarity", "Sentiment_Subjectivity"]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")

    mask_all_nan = df[cols].isna().all(axis=1)
    n_dropped_all4 = int(mask_all_nan.sum())
    df_clean = df.loc[~mask_all_nan].copy()
    return df_clean, n_dropped_all4


def drop_rows_with_empty_text(df: pd.DataFrame,
                              text_col: str = "Translated_Review") -> tuple[pd.DataFrame, int]:
    """Drop rows with empty or NA text.

    Args:
      df: DataFrame to clean.
      text_col: Text column name.

    Returns:
      (df2, n_drop)
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing expected column: {text_col}")

    mask_empty = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
    n_drop = int(mask_empty.sum())
    df2 = df.loc[~mask_empty].copy()
    return df2, n_drop


def run_reviews_cleaning(in_csv: str | Path,
                         out_csv: str | Path,
                         also_drop_empty_text: bool = True) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Clean the reviews CSV and save a single final file.

    Steps:
      1) Drop rows where all four (text + sentiment fields) are NA.
      2) Normalize sentiment labels to {Positive, Negative, Neutral}.
      3) Optionally drop empty text rows.
      4) Save to `out_csv`.

    Args:
      in_csv: Input CSV path.
      out_csv: Output CSV path.
      also_drop_empty_text: Whether to drop empty text rows.

    Returns:
      (df, report) where report has:
        - dropped_all4
        - changed_sentiment
        - dropped_empty_text
        - final_rows
        - final_cols
    """
    df, n_dropped_all4 = drop_all_nan_reviews(in_csv)
    df, n_changed = normalize_sentiment_labels(df, col="Sentiment")

    n_drop_txt = 0
    if also_drop_empty_text:
        df, n_drop_txt = drop_rows_with_empty_text(df, text_col="Translated_Review")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    report = {
        "dropped_all4": n_dropped_all4,
        "changed_sentiment": n_changed,
        "dropped_empty_text": n_drop_txt,
        "final_rows": int(df.shape[0]),
        "final_cols": int(df.shape[1]),
    }
    return df, report


def _compute_basic_stats(
    df: pd.DataFrame,
    cols: list[str],
    ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Compute min/max/mean/std and out-of-range counts for numeric columns.

    Args:
      df: DataFrame.
      cols: Columns to summarize.
      ranges: Valid ranges per column.

    Returns:
      Stats DataFrame.
    """
    rows: List[Dict[str, float | int | str]] = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        count_non_null = int(s.notna().sum())
        count_na = int(s.isna().sum())
        mn = float(s.min()) if count_non_null else float("nan")
        mx = float(s.max()) if count_non_null else float("nan")
        mean = float(s.mean()) if count_non_null else float("nan")
        std = float(s.std(ddof=1)) if count_non_null > 1 else float("nan")
        lo, hi = ranges.get(c, (-float("inf"), float("inf")))
        out_range = int(((s < lo) | (s > hi)).sum())

        rows.append({
            "column": c,
            "count_non_null": count_non_null,
            "count_na": count_na,
            "min": mn,
            "max": mx,
            "mean": mean,
            "std": std,
            "out_of_range": out_range,
        })

    return pd.DataFrame(rows)


def _plot_histograms(df: pd.DataFrame,
                     cols: list[str],
                     out_dir: Path,
                     bins: int = 30) -> None:
    """Save histograms for given numeric columns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        plt.figure()
        plt.hist(s, bins=bins)
        plt.title(f"Histogram of {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.savefig(out_dir / f"hist_{c}.png", bbox_inches="tight")
        plt.close()


def _plot_scatter(df: pd.DataFrame,
                  x_col: str,
                  y_col: str,
                  out_path: Path) -> None:
    """Save a scatter figure for x_col vs y_col."""
    x = pd.to_numeric(df.get(x_col, pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(df.get(y_col, pd.Series(dtype=float)), errors="coerce")
    mask = x.notna() & y.notna()
    x2, y2 = x[mask], y[mask]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(x2, y2, s=6, alpha=0.6)
    plt.title(f"{x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def run_reviews_stats(in_csv: str | Path,
                      out_dir: str | Path) -> pd.DataFrame:
    """Compute basic stats and save figures for the cleaned reviews file.

    Generates:
      - histograms for Sentiment_Polarity and Sentiment_Subjectivity
      - scatter plot (polarity vs subjectivity)
      - returns a stats DataFrame

    Args:
      in_csv: Cleaned CSV path (output of run_reviews_cleaning).
      out_dir: Figure output directory.

    Returns:
      Stats DataFrame with min/max/mean/std, NA counts, and out-of-range counts.
    """
    df = pd.read_csv(in_csv)

    cols = ["Sentiment_Polarity", "Sentiment_Subjectivity"]
    ranges = {
        "Sentiment_Polarity": (-1.0, 1.0),
        "Sentiment_Subjectivity": (0.0, 1.0),
    }

    stats_df = _compute_basic_stats(df, cols=cols, ranges=ranges)

    out_dir = Path(out_dir)
    _plot_histograms(df, cols=cols, out_dir=out_dir, bins=30)
    _plot_scatter(
        df,
        x_col="Sentiment_Polarity",
        y_col="Sentiment_Subjectivity",
        out_path=out_dir / "scatter_polarity_subjectivity.png",
    )
    return stats_df
