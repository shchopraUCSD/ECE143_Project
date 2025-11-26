from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_installs_by_last_updated(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/app_active_dev"
) -> None:
    """Plot box plot of Installs by Last Updated month."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Last Updated"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Last Updated"] = pd.to_datetime(df_subset["Last Updated"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset["Installs"] > 0]
    
    df_subset["YearMonth"] = df_subset["Last Updated"].dt.to_period("M")
    df_subset = df_subset.sort_values("YearMonth")
    
    plt.figure(figsize=(20, 8))
    unique_months = df_subset["YearMonth"].unique()
    data_by_month = [df_subset[df_subset["YearMonth"] == month]["Installs"].values for month in unique_months]
    
    bp = plt.boxplot(data_by_month, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    
    medians = [np.median(data) for data in data_by_month]
    x_positions = range(1, len(medians) + 1)
    plt.plot(x_positions, medians, 'r:', linewidth=3, alpha=0.8)
    
    plt.xlabel("Last Updated (Year-Month)", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs by Last Updated Month", fontsize=14, pad=20)
    plt.xticks([])
    plt.yscale("log")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_by_last_updated.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Box plot saved to: {output_file}")


def plot_installs_by_android_ver(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/app_active_dev"
) -> None:
    """Plot box plot of Installs by Android Ver."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Android Ver"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Android Ver"] = pd.to_numeric(df_subset["Android Ver"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset["Installs"] > 0]
    
    android_vers = sorted(df_subset["Android Ver"].unique())
    data_by_ver = [df_subset[df_subset["Android Ver"] == ver]["Installs"].values for ver in android_vers]
    
    plt.figure(figsize=(12, 8))
    bp = plt.boxplot(data_by_ver, tick_labels=[str(int(ver)) for ver in android_vers], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    
    medians = [np.median(data) for data in data_by_ver]
    x_positions = range(1, len(medians) + 1)
    plt.plot(x_positions, medians, 'r:', linewidth=3, alpha=0.8)
    
    plt.xlabel("Android Ver", fontsize=12)
    plt.ylabel("Installs", fontsize=12)
    plt.title("Installs by Android Ver", fontsize=14, pad=20)
    plt.yscale("log")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_by_android_ver.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Box plot saved to: {output_file}")


def plot_installs_heatmap_by_activity(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/app_active_dev"
) -> None:
    """Plot 2D heatmap of average Installs by Last Updated and Android Ver quartiles."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df_subset = df[["Installs", "Last Updated", "Android Ver"]].copy()
    df_subset["Installs"] = pd.to_numeric(df_subset["Installs"], errors="coerce")
    df_subset["Last Updated"] = pd.to_datetime(df_subset["Last Updated"], errors="coerce")
    df_subset["Android Ver"] = pd.to_numeric(df_subset["Android Ver"], errors="coerce")
    df_subset = df_subset.dropna()
    df_subset = df_subset[df_subset["Installs"] > 0]
    
    df_subset["LastUpdated_quartile"] = pd.qcut(df_subset["Last Updated"].rank(method="first"), q=4, labels=["Q1 (Oldest)", "Q2", "Q3", "Q4 (Newest)"])
    df_subset["AndroidVer_quartile"] = pd.qcut(df_subset["Android Ver"].rank(method="first"), q=4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])
    
    heatmap_data = df_subset.groupby(["LastUpdated_quartile", "AndroidVer_quartile"], observed=True)["Installs"].mean().unstack()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Average Installs"})
    plt.xlabel("Android Ver Quartile", fontsize=12)
    plt.ylabel("Last Updated Quartile", fontsize=12)
    plt.title("Average Installs by Last Updated and Android Ver Quartiles", fontsize=14, pad=20)
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "installs_heatmap_by_activity.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Heatmap saved to: {output_file}")


if __name__ == "__main__":
    plot_installs_by_last_updated()
    plot_installs_by_android_ver()
    plot_installs_heatmap_by_activity()

