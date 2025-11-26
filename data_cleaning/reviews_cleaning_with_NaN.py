from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = Path("data_raw/googleplaystore_user_reviews.csv")
OUTPUT_DIR = Path('data_processed')

def normalize_columns(df):
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].str.title()

    for col in ["Sentiment_Polarity", "Sentiment_Subjectivity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def build_stats(df):

    required = ["App", "Translated_Review", "Sentiment", "Sentiment_Polarity", "Sentiment_Subjectivity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    def review_list(series):
        seen = set()
        ordered_unique = []
        for x in series.dropna():
            if x not in seen:
                seen.add(x)
                ordered_unique.append(x)
        return " || ".join(ordered_unique)

    def positive_percent(group):
        s = group["Sentiment"].dropna()
        denom = len(s)
        if denom == 0:
            return np.nan
        numer = (s == "Positive").sum()
        return float(numer) / float(denom)

    def count_nans(group):
        # Count NaNs in Translated_Review column only
        return int(group["Translated_Review"].isna().sum())

    grouped = df.groupby("App", dropna=False)

    # Calculate mean, median and std without Nan
    out = pd.DataFrame({
        "Translated_Review_list": grouped["Translated_Review"].apply(review_list),
        "Sentiment_Positive_Percentile": grouped.apply(positive_percent, include_groups=False),
        "Sentiment_Polarity_Mean": grouped["Sentiment_Polarity"].mean(),
        "Sentiment_Polarity_Median": grouped["Sentiment_Polarity"].median(),
        "Sentiment_Polarity_STD": grouped["Sentiment_Polarity"].std(),
        "Sentiment_Subjectivity_Mean": grouped["Sentiment_Subjectivity"].mean(),
        "Sentiment_Subjectivity_Median": grouped["Sentiment_Subjectivity"].median(),
        "Sentiment_Subjectivity_STD": grouped["Sentiment_Subjectivity"].std(),
        "Number_of_Nans": grouped.apply(count_nans, include_groups=False),
        "Total_Reviews": grouped.size(),  # Total number of reviews including NaNs
    }).reset_index()

    return out

def main():
    df = pd.read_csv(INPUT_FILE)
    df = normalize_columns(df)
    cleaned_path = OUTPUT_DIR / f"{INPUT_FILE.stem}_cleaned_keep_nans.csv"
    df.to_csv(cleaned_path, index=False)
    stats_df = build_stats(df)
    stats_path = OUTPUT_DIR / f"{INPUT_FILE.stem}_app_stats.csv"
    stats_df.to_csv(stats_path, index=False)
if __name__ == "__main__":
    main()