import pandas as pd
import matplotlib.pyplot as plt


def count_nan_rating_low_installs(df, low_thresh):
    """
    Count apps with NaN rating and low installs.
    """
    nan_rating_df = df[df['Rating'].isna()]
    low_installs = nan_rating_df[nan_rating_df['Installs'] <= low_thresh]
    return len(low_installs)


def count_nan_rating_high_installs(df, high_thresh):
    """
    Count apps with NaN rating and high installs.
    """
    nan_rating_df = df[df['Rating'].isna()]
    high_installs = nan_rating_df[nan_rating_df['Installs'] >= high_thresh]
    return len(high_installs)


def plot_nan_rating_pie_all(df, low_thresh=1000, high_thresh=100000):
    """
    Pie chart of NaN rating apps relative to all apps.
    """
    total_apps = len(df)
    low_count = count_nan_rating_low_installs(df, low_thresh)
    high_count = count_nan_rating_high_installs(df, high_thresh)
    other_count = total_apps - low_count - high_count

    labels = [
        "NaN rating, low installs",
        "NaN rating, high installs",
        "Other apps",
    ]
    sizes = [low_count, high_count, other_count]
    colors = ["#F1920B", "#197CEC", "#BEBCBC"]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct="%.2f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("NaN ratings by install level", fontsize=12)
    ax.axis("equal")

    legend_labels = [
        f"{labels[0]} (n={low_count})",
        f"{labels[1]} (n={high_count})",
        f"{labels[2]} (n={other_count})",
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )

    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("data_processed/googleplaystore_clean.csv")

    low_count = count_nan_rating_low_installs(df, 1000)
    high_count = count_nan_rating_high_installs(df, 100000)
    total_apps = len(df)

    print(f"Apps with NaN rating and low installs: {low_count}")
    print(f"Apps with NaN rating and high installs: {high_count}")
    print(f"Total apps: {total_apps}")

    plot_nan_rating_pie_all(df, low_thresh=1000, high_thresh=100000)


if __name__ == "__main__":
    main()
