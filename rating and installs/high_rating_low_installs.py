import pandas as pd
import matplotlib.pyplot as plt


def get_high_rating_low_installs(df, rating_thresh=4.5, install_thresh=1000):
    """
    Return subset of apps with high rating and low installs.
    """
    df = df.dropna(subset=["Rating", "Installs"])
    mask = (df["Rating"] >= rating_thresh) & (df["Installs"] <= install_thresh)
    return df[mask]


def plot_high_rating_low_installs(df, rating_thresh=4.5, install_thresh=1000):
    """
    Scatter plot highlighting high rating and low installs apps.
    """
    sub = df.dropna(subset=["Rating", "Installs"])
    highlighted = get_high_rating_low_installs(sub, rating_thresh, install_thresh)

    fig, ax = plt.subplots()

    ax.scatter(
        sub["Installs"],
        sub["Rating"],
        alpha=0.2,
        s=10,
        label=f"All apps (n={len(sub)})",
    )
    ax.scatter(
        highlighted["Installs"],
        highlighted["Rating"],
        alpha=0.8,
        s=25,
        label=f"High rating & low installs (n={len(highlighted)})",
    )

    ax.axhline(rating_thresh, linestyle="--")
    ax.axvline(install_thresh, linestyle="--")

    ax.set_xscale("log")
    ax.set_xlabel("Installs (log scale)")
    ax.set_ylabel("Rating")
    ax.set_title("High rating but low installs")

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_high_rating_low_installs_by_month(df, rating_thresh=4.5, install_thresh=1000):
    """
    Bar chart of high rating, low install apps by last updated month,
    with different colors for different years.
    """
    subset = get_high_rating_low_installs(df, rating_thresh, install_thresh).copy()
    subset["LastUpdated_dt"] = pd.to_datetime(subset["Last Updated"], errors="coerce")
    subset = subset.dropna(subset=["LastUpdated_dt"])

    subset["year_month"] = subset["LastUpdated_dt"].dt.to_period("M").astype(str)
    month_counts = subset["year_month"].value_counts().sort_index()

    months = list(month_counts.index)
    years = [m[:4] for m in months]
    unique_years = sorted(set(years))

    cmap = plt.get_cmap("tab10", len(unique_years))
    colors = [cmap(unique_years.index(y)) for y in years]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(month_counts)), month_counts.values, color=colors)

    ax.set_xlabel("Last Updated (year-month)")
    ax.set_ylabel("Number of apps")
    ax.set_title("High rating & low installs by last updated month")

    step = 6
    tick_idx = list(range(0, len(month_counts), step))
    last_idx = len(month_counts) - 1
    if last_idx not in tick_idx:
        tick_idx.append(last_idx)
    tick_idx = sorted(tick_idx)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([months[i] for i in tick_idx], rotation=45, ha="right")

    handles = []
    labels = []
    for i, y in enumerate(unique_years):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=cmap(i),
                marker="s",
                linestyle="",
            )
        )
        labels.append(y)
    ax.legend(handles, labels, title="Year", loc="upper left")

    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("data_processed/googleplaystore_clean.csv")

    rating_thresh = 4.5
    install_thresh = 1000

    plot_high_rating_low_installs(
        df, rating_thresh=rating_thresh, install_thresh=install_thresh
    )
    plot_high_rating_low_installs_by_month(
        df, rating_thresh=rating_thresh, install_thresh=install_thresh
    )


if __name__ == "__main__":
    main()
