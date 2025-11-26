import pandas as pd
import matplotlib.pyplot as plt


def count_high_rating_low_installs(df, rating_thresh, install_thresh):
    """
    Return number of apps with high rating and low installs.
    """
    df = df.dropna(subset=['Rating', 'Installs'])
    mask = (df['Rating'] >= rating_thresh) & (df['Installs'] <= install_thresh)
    result = df[mask]
    return len(result)


def plot_high_rating_low_installs(df, rating_thresh=4.0, install_thresh=10_000):
    """
    Scatter plot highlighting high rating and low installs apps.
    """
    sub = df.dropna(subset=['Rating', 'Installs'])
    mask = (sub['Rating'] >= rating_thresh) & (sub['Installs'] <= install_thresh)
    highlighted = sub[mask]

    fig, ax = plt.subplots()
    ax.scatter(sub['Installs'], sub['Rating'], alpha=0.2, s=10, label='All apps')
    ax.scatter(
        highlighted['Installs'],
        highlighted['Rating'],
        alpha=0.8,
        s=25,
        label='High rating & low installs'
    )

    ax.axhline(rating_thresh, linestyle='--')
    ax.axvline(install_thresh, linestyle='--')

    ax.set_xscale('log')
    ax.set_xlabel('Installs (log scale)')
    ax.set_ylabel('Rating')
    ax.set_title('High rating but low installs')

    ax.text(
        0.05,
        0.95,
        f'N = {len(highlighted)}',
        transform=ax.transAxes,
        ha='left',
        va='top'
    )

    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv('data_processed/googleplaystore_clean.csv')

    rating_thresh = 4.5
    install_thresh = 1000

    high_rating_low_installs_count = count_high_rating_low_installs(
        df, rating_thresh, install_thresh
    )

    total_apps = len(df)
    high_rating_apps = df[df['Rating'] >= rating_thresh]
    high_rating_count = len(high_rating_apps)

    pct_within_high_rating = (
        high_rating_low_installs_count / high_rating_count * 100
        if high_rating_count > 0 else 0.0
    )
    pct_of_all = (
        high_rating_low_installs_count / total_apps * 100
        if total_apps > 0 else 0.0
    )

    print(f"Apps with high rating and low installs: {high_rating_low_installs_count}")
    print(
        f"Among high rating apps (>= {rating_thresh}), "
        f"{pct_within_high_rating:.2f}% have installs <= {install_thresh}."
    )
    print(
        f"High rating, low-install apps account for {pct_of_all:.2f}% "
        f"of all apps in the dataset."
    )

    plot_high_rating_low_installs(
        df, rating_thresh=rating_thresh, install_thresh=install_thresh
    )


if __name__ == "__main__":
    main()
