# main.py
from reviews_cleaning import run_reviews_cleaning, run_reviews_stats
from apps_cleaning import head_with_csv_lines, clean_googleplay_apps, apps_basic_stats


def _sample_lines(lines, k=20):
    return lines[:k] if lines else []


if __name__ == "__main__":
    # ======= Reviews: clean + stats + figs =======
    IN_REV  = "data_raw/googleplaystore_user_reviews.csv"
    OUT_REV = "data_processed/reviews_clean.csv"
    FIG_REV = "figs/reviews"

    df_rev, rep_rev = run_reviews_cleaning(IN_REV, out_csv=OUT_REV, also_drop_empty_text=True)
    stats_rev = run_reviews_stats(OUT_REV, out_dir=FIG_REV)

    print(f"Final shape: ({rep_rev['final_rows']}, {rep_rev['final_cols']})")
    print(f"Dropped (ALL FOUR missing): {rep_rev['dropped_all4']}")
    print(f"Sentiment normalized (changed rows): {rep_rev['changed_sentiment']}")
    print(f"Dropped empty text rows: {rep_rev['dropped_empty_text']}")
    print("\n[Reviews] Basic stats:")
    print(stats_rev.to_string(index=False))
    print(f"\nFigures saved to: {FIG_REV}")

    # ======= Apps: preview first 5 rows in CSV order =======
    IN_APPS = "data_raw/googleplaystore.csv"
    preview = head_with_csv_lines(IN_APPS, n=5)
    print("\nFirst 5 data rows with CSV line numbers:")
    print(preview.to_string(index=False))

    # ======= Apps: clean (keep latest per App) =======
    OUT_APPS = "data_processed/googleplaystore_clean.csv"
    df_apps, rep_apps = clean_googleplay_apps(
        IN_APPS, out_csv=OUT_APPS, keep_latest_per_app=True
    )

    print("\n=== Apps row accounting ===")
    print(
        "Rows in: {rin}  |  Final rows: {rout}  |  Exact-dup rows dropped: {d1}  |  'App'-dup rows dropped: {d2}"
        .format(rin=rep_apps.get("rows_in", 0),
                rout=rep_apps.get("rows_out", 0),
                d1=rep_apps.get("dup_rows_dropped", 0),
                d2=rep_apps.get("dup_apps_dropped", 0))
    )

    print(f"\nSaved cleaned file → {OUT_APPS}")
    print(f"Final shape: ({rep_apps.get('rows_out', len(df_apps))}, {df_apps.shape[1]})")

    # Show anomaly counts + sample CSV lines
    print(f"Exact-dup rows dropped: {rep_apps.get('dup_rows_dropped', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("dup_rows_lines", [])))

    print(f"Duplicate Apps dropped: {rep_apps.get('dup_apps_dropped', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("dup_apps_lines", [])))

    print(f"Rating out-of-range coerced to NA: {rep_apps.get('rating_out_of_range', 0)}")
    print("  -> bad csv lines (sample):", _sample_lines(rep_apps.get("rating_bad_lines", [])))

    print(f"Installs NA: {rep_apps.get('installs_na', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("installs_na_lines", [])))

    print(f"Price NA: {rep_apps.get('price_na', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("price_na_lines", [])))

    print(f"Size NA: {rep_apps.get('size_na', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("size_na_lines", [])))

    print(f"Type NA: {rep_apps.get('type_na', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("type_na_lines", [])))

    print(f"Current Ver NA: {rep_apps.get('current_ver_na', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("current_ver_na", [])))

    print(f"Android Ver NA: {rep_apps.get('android_ver_na', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("android_ver_na", [])))

    print(f"Genres NA: {rep_apps.get('genres_na', 0)}")
    print("  -> csv lines (sample):", _sample_lines(rep_apps.get("genres_na", [])))


    # ======= Apps: numeric basic stats  =======
    stats_apps = apps_basic_stats(df_apps)
    print("\n[Apps] Basic stats:")
    print(stats_apps.to_string(index=False))
