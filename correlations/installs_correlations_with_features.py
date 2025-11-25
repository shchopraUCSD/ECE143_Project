from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_installs_correlations_heatmap(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot correlation heatmap between Installs and features."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Rating", "Reviews", "Price", "Size", "Android Ver", "Installs"]].copy()
    
    df_subset["Android_Ver_encoded"] = pd.Categorical(df_subset["Android Ver"]).codes
    df_subset["Android_Ver_encoded"] = df_subset["Android_Ver_encoded"].replace(-1, np.nan)
    
    corr_columns = ["Rating", "Reviews", "Price", "Size", "Android_Ver_encoded", "Installs"]
    df_corr = df_subset[corr_columns].copy()
    
    for col in df_corr.columns:
        df_corr[col] = pd.to_numeric(df_corr[col], errors="coerce")
    
    corr_matrix = df_corr.corr()
    installs_corr = corr_matrix["Installs"].drop("Installs").sort_values(ascending=False)
    
    name_mapping = {
        "Android_Ver_encoded": "Android Ver"
    }
    feature_names = [name_mapping.get(name, name) for name in installs_corr.index.tolist()]
    
    heatmap_data = installs_corr.values.reshape(1, -1)
    
    vmin, vmax = -0.1, 0.1
    
    plt.figure(figsize=(12, 2))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=feature_names,
        yticklabels=["Installs"],
        cbar_kws={"label": "Correlation Coefficient"}
    )
    plt.title("Correlation Heatmap: Installs vs Features", fontsize=14, pad=20)
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_correlations_heatmap.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Heatmap saved to: {output_file}")
    print("\nCorrelation values with Installs:")
    print(installs_corr.to_string())


def plot_installs_vs_reviews_scatter(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot scatter plot of Installs vs Reviews with log scales."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Reviews"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Reviews"] = pd.to_numeric(df_subset["Reviews"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[(df_subset["Installs"] > 0) & (df_subset["Reviews"] > 0)]
    
    log_reviews = np.log10(df_subset["Reviews"])
    log_installs = np.log10(df_subset["Installs"])
    
    slope, intercept = np.polyfit(log_reviews, log_installs, 1)
    
    x_line = np.logspace(log_reviews.min(), log_reviews.max(), 100)
    y_line = 10**(slope * np.log10(x_line) + intercept)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(df_subset["Reviews"], df_subset["Installs"], alpha=0.5, s=10)
    plt.plot(x_line, y_line, 'r:', linewidth=3, alpha=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Reviews", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs vs Reviews (Log Scale)", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_vs_reviews_scatter_log.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Scatter plot saved to: {output_file}")


def plot_installs_vs_reviews_scatter_linear(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot scatter plot of Installs vs Reviews with linear scales."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Reviews"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Reviews"] = pd.to_numeric(df_subset["Reviews"], errors="coerce")
    df_subset = df_subset.dropna()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(df_subset["Reviews"], df_subset["Installs"], alpha=0.5, s=10)
    plt.xlabel("Reviews", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs vs Reviews (Linear Scale)", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_vs_reviews_scatter_linear.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Scatter plot (linear) saved to: {output_file}")


def plot_installs_vs_price_scatter(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot scatter plot of Installs vs Price with y-axis log scale."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Price"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Price"] = pd.to_numeric(df_subset["Price"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset["Installs"] > 0]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(df_subset["Price"], df_subset["Installs"], alpha=0.5, s=10)
    plt.yscale("log")
    plt.xlabel("Price", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs vs Price (Y-axis Log Scale)", fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_vs_price_scatter_log.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Scatter plot saved to: {output_file}")


def plot_installs_by_category_boxplot(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot box plot of Installs by Category."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Category", "Installs"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset["Installs"] > 0]
    
    median_installs = df_subset.groupby("Category")["Installs"].median().sort_values(ascending=False)
    categories = median_installs.index.tolist()
    
    norm = plt.Normalize(vmin=median_installs.min(), vmax=median_installs.max())
    cmap = plt.cm.get_cmap("YlOrRd")
    
    data_by_category = [df_subset[df_subset["Category"] == cat]["Installs"].values for cat in categories]
    colors = [cmap(norm(median_installs[cat])) for cat in categories]
    
    plt.figure(figsize=(16, 8))
    bp = plt.boxplot(data_by_category, labels=categories, patch_artist=True)
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs by Category (Colored by Median Installs)", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_by_category_boxplot.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Box plot saved to: {output_file}")


if __name__ == "__main__":
    plot_installs_correlations_heatmap()
    plot_installs_vs_reviews_scatter()
    plot_installs_vs_reviews_scatter_linear()
    plot_installs_vs_price_scatter()
    plot_installs_by_category_boxplot()
