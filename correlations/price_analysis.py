from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table


def print_apps_price_over_200(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Print apps with price > 200."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    high_price_apps = df[df["Price"] > 200].sort_values("Price", ascending=False)
    
    if len(high_price_apps) > 0:
        print(f"Apps with price > 200 ({len(high_price_apps)} apps):\n")
        print(high_price_apps.to_string(index=False))
        
        top5 = high_price_apps.head(5)[["Price", "App", "Category", "Rating", "Reviews", "Installs"]].copy()
        top5["Price"] = top5["Price"].apply(lambda x: f"${x:.2f}")
        
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.axis("tight")
        ax.axis("off")
        
        table = ax.table(
            cellText=top5.values,
            colLabels=top5.columns,
            cellLoc="left",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(len(top5.columns)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")
        
        plt.title("Top 5 Priciest Apps (Price > $200)", fontsize=14, pad=20, weight="bold")
        
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "top5_priciest_apps.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()
        
        print(f"\nImage saved to: {output_file}")
    else:
        print("No apps found with price > 200")


def plot_avg_installs_by_type(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_dir: str = "figs/correlations"
) -> None:
    """Plot bar graph of average installs by Type."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    output_path = root / output_dir
    
    df = pd.read_csv(csv_file)
    
    df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")
    avg_installs = df.groupby("Type")["Installs"].mean()
    
    plt.figure(figsize=(8, 6))
    avg_installs.plot(kind="bar", color=["#4CAF50", "#2196F3"])
    plt.xlabel("Type", fontsize=12)
    plt.ylabel("Average Installs", fontsize=12)
    plt.title("Average Installs by App Type", fontsize=14, pad=20, weight="bold")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "avg_installs_by_type.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Bar graph saved to: {output_file}")


if __name__ == "__main__":
    print_apps_price_over_200()
    plot_avg_installs_by_type()

