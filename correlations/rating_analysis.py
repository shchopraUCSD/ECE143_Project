from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_installs_vs_rating(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot histogram of Installs vs Rating with 0.1 bands."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Rating"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Rating"] = pd.to_numeric(df_subset["Rating"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset["Installs"] > 0]
    
    bins = np.arange(0, 5.1, 0.1)
    df_subset["Rating_band"] = pd.cut(df_subset["Rating"], bins=bins, include_lowest=True)
    installs_by_band = df_subset.groupby("Rating_band", observed=True)["Installs"].sum()
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(installs_by_band)), installs_by_band.values, edgecolor="black", alpha=0.7, width=0.8)
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs by Rating (0.1 bands)", fontsize=14, pad=20)
    
    rating_positions = []
    rating_labels = []
    for rating in [1, 2, 3, 4, 5]:
        for i, interval in enumerate(installs_by_band.index):
            if interval.left <= rating < interval.right or (rating == 5.0 and interval.right == 5.1):
                rating_positions.append(i)
                rating_labels.append(str(rating))
                break
    
    plt.xticks(rating_positions, rating_labels)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_vs_rating.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Histogram saved to: {output_file}")


if __name__ == "__main__":
    plot_installs_vs_rating()

