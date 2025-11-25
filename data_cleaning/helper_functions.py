from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

