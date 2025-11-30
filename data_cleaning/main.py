from apps_cleaning import clean_googleplay_apps, apps_basic_stats

if __name__ == "__main__":
    FIG_REV = "figs/reviews"
    IN_APPS = "data_raw/googleplaystore.csv"
    OUT_APPS = "data_processed/googleplaystore_clean.csv"
    df_apps, rep_apps = clean_googleplay_apps(
        IN_APPS, out_csv=OUT_APPS, keep_latest_per_app=True
    )
    stats_apps = apps_basic_stats(df_apps)
    print(stats_apps.to_string(index=False))
