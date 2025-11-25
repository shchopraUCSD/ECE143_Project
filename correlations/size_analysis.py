from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_installs_vs_size_by_category(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot scatter plot of Installs vs Size colored by Category."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Size", "Category"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Size"] = pd.to_numeric(df_subset["Size"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset["Installs"] > 0]
    
    top5_categories = df_subset["Category"].value_counts().head(5).index.tolist()
    df_subset["Category_group"] = df_subset["Category"].apply(
        lambda x: x if x in top5_categories else "Other"
    )
    
    categories_ordered = top5_categories + ["Other"]
    distinct_colors = ["#FF0000", "#00FF00", "#FFFF00", "#87CEEB", "#FF69B4", "#D3D3D3"]
    color_map = dict(zip(categories_ordered, distinct_colors[:len(categories_ordered)]))
    
    plt.figure(figsize=(12, 8))
    for category in categories_ordered:
        mask = df_subset["Category_group"] == category
        plt.scatter(
            df_subset.loc[mask, "Size"],
            df_subset.loc[mask, "Installs"],
            c=[color_map[category]],
            label=category,
            alpha=0.6,
            s=10
        )
    
    plt.xlabel("Size (MB)", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs vs Size (Colored by Category)", fontsize=14, pad=20)
    plt.yscale("log")
    plt.legend(title="Category", fontsize=9, title_fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_vs_size_by_category.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Scatter plot saved to: {output_file}")


if __name__ == "__main__":
    plot_installs_vs_size_by_category()

