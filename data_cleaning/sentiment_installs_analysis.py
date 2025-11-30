"""
Analysis of sentiment metrics correlation with installs and subjectivity patterns.
Analyzes:
1. Correlation between sentiment metrics and number of installs
2. Objectivity vs subjectivity in comments
3. Subjectivity patterns across categories
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# File paths
APPS_FILE = Path("data_processed/googleplaystore_clean.csv")
REVIEWS_FILE = Path("data_processed/googleplaystore_user_reviews_app_stats.csv")
OUTPUT_DIR = Path("analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_merge_data():
    """Load and merge apps and reviews data."""
    apps = pd.read_csv(APPS_FILE)
    reviews = pd.read_csv(REVIEWS_FILE)
    # Merge on App name
    merged = apps.merge(reviews, on='App', how='inner')
    print(f"Merged data: {len(merged)} rows")
    return merged


def analyze_correlations(df):
    """Analyze correlations between sentiment metrics and installs."""
    print("Sentiment Metrics vs Installs")
    # relevant columns for correlation
    correlation_cols = [
        'Installs',
        'Total_Reviews',
        'Sentiment_Positive_Percentile',
        'Sentiment_Polarity_Mean',
        'Sentiment_Subjectivity_Mean',
        'Sentiment_Polarity_STD',
        'Sentiment_Subjectivity_STD'
    ]

    corr_df = df[correlation_cols].copy()
    corr_matrix = corr_df.corr()
    # Print correlations with Installs
    print("\nCorrelations with Installs:")
    installs_corr = corr_matrix['Installs'].sort_values(ascending=False)
    for col, corr_val in installs_corr.items():
        if col != 'Installs':
            print(f"{col:40s}: {corr_val:7.4f}")

    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                cmap='coolwarm', center=0, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Sentiment Metrics and Installs',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n Saved: {OUTPUT_DIR / 'correlation_heatmap.png'}")
    plt.close()

    # Scatter plots for key relationships
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1 Installs vs Total Reviews
    ax = axes[0, 0]
    ax.scatter(df['Total_Reviews'], df['Installs'], alpha=0.5, s=30)
    ax.set_xlabel('Total Reviews (including NaNs)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Installs', fontsize=11, fontweight='bold')
    ax.set_title('Installs vs Total Reviews', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add regression line
    valid_mask = (df['Total_Reviews'] > 0) & (df['Installs'] > 0)
    if valid_mask.sum() > 0:
        log_reviews = np.log10(df.loc[valid_mask, 'Total_Reviews'])
        log_installs = np.log10(df.loc[valid_mask, 'Installs'])
        z = np.polyfit(log_reviews, log_installs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(log_reviews.min(), log_reviews.max(), 100)
        ax.plot(10**x_line, 10**p(x_line), "r--", linewidth=2, alpha=0.8,
                label=f'r={installs_corr["Total_Reviews"]:.3f}')
        ax.legend()

    # 2. Installs vs Sentiment Positive Percentile
    ax = axes[0, 1]
    ax.scatter(df['Sentiment_Positive_Percentile'], df['Installs'], alpha=0.5, s=30)
    ax.set_xlabel('Sentiment Positive Percentile', fontsize=11, fontweight='bold')
    ax.set_ylabel('Installs', fontsize=11, fontweight='bold')
    ax.set_title('Installs vs Sentiment Positive %', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()

    # 3. Installs vs Sentiment Subjectivity Mean
    ax = axes[1, 0]
    ax.scatter(df['Sentiment_Subjectivity_Mean'], df['Installs'], alpha=0.5, s=30)
    ax.set_xlabel('Sentiment Subjectivity Mean', fontsize=11, fontweight='bold')
    ax.set_ylabel('Installs', fontsize=11, fontweight='bold')
    ax.set_title('Installs vs Sentiment Subjectivity', fontsize=12, fontweight='bold')
    ax.set_yscale('log')

    # 4. Installs vs Sentiment Polarity Mean
    ax = axes[1, 1]
    ax.scatter(df['Sentiment_Polarity_Mean'], df['Installs'], alpha=0.5, s=30)
    ax.set_xlabel('Sentiment Polarity Mean', fontsize=11, fontweight='bold')
    ax.set_ylabel('Installs', fontsize=11, fontweight='bold')
    ax.set_title('Installs vs Sentiment Polarity', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Neutral')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'installs_vs_sentiment_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'installs_vs_sentiment_scatter.png'}")
    plt.close()

    return corr_matrix


def analyze_objectivity_vs_subjectivity(df):
    """Analyze whether comments tend to be more objective or subjective."""
    print("Objectivity vs Subjectivity Analysis")
    subjectivity = df['Sentiment_Subjectivity_Mean'].dropna()
    print(f"\nSubjectivity Statistics:")
    print(f"Mean:{subjectivity.mean():.4f}")
    print(f"Median:{subjectivity.median():.4f}")
    print(f"Std:{subjectivity.std():.4f}")
    print(f"Min:{subjectivity.min():.4f}")
    print(f"Max:{subjectivity.max():.4f}")
    objective_count = (subjectivity < 0.5).sum()
    subjective_count = (subjectivity >= 0.5).sum()


    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Distribution histogram
    ax = axes[0]
    ax.hist(subjectivity, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(subjectivity.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {subjectivity.mean():.3f}')
    ax.axvline(subjectivity.median(), color='green', linestyle='--', linewidth=2,
               label=f'Median: {subjectivity.median():.3f}')
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=2,
               label='Objective/Subjective Threshold')
    ax.set_xlabel('Subjectivity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Apps', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Comment Subjectivity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.05, 0.05)
    ax.pie([objective_count, subjective_count],
           labels=['More Objective\n(< 0.5)', 'More Subjective\n(≥ 0.5)'],
           autopct='%1.1f%%', startangle=90, colors=colors, explode=explode,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Objective vs Subjective Comments', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'objectivity_vs_subjectivity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'objectivity_vs_subjectivity.png'}")
    plt.close()


def analyze_subjectivity_by_category(df):
    """Analyze relationship between subjectivity and app categories."""
    print("Subjectivity by Category Analysis")

    # Calculate mean subjectivity by category
    category_subjectivity = df.groupby('Category').agg({
        'Sentiment_Subjectivity_Mean': ['mean', 'median', 'std', 'count']
    }).round(4)

    category_subjectivity.columns = ['Mean', 'Median', 'Std', 'Count']
    category_subjectivity = category_subjectivity.sort_values('Mean', ascending=False)
    print("\nTop 10 Most Subjective Categories:")
    print(category_subjectivity.head(10).to_string())
    print("\nTop 10 Most Objective Categories:")
    print(category_subjectivity.tail(10).to_string())

    # Filter categories with at least 10 apps for visualization
    categories_filtered = category_subjectivity[category_subjectivity['Count'] >= 10].copy()

    # Create visualizations
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # 1. Bar plot of mean subjectivity by category
    ax = axes[0]
    categories_sorted = categories_filtered.sort_values('Mean', ascending=True)
    y_pos = np.arange(len(categories_sorted))

    colors = ['green' if x < 0.5 else 'red' for x in categories_sorted['Mean']]

    ax.barh(y_pos, categories_sorted['Mean'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories_sorted.index, fontsize=9)
    ax.set_xlabel('Mean Subjectivity Score', fontsize=12, fontweight='bold')
    ax.set_title('Mean Comment Subjectivity by Category (>10 apps)',
                 fontsize=14, fontweight='bold')
    ax.axvline(0.5, color='blue', linestyle='--', linewidth=2, alpha=0.7,
               label='Objective/Subjective Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Box plot for top categories by count
    ax = axes[1]
    top_categories = category_subjectivity.nlargest(15, 'Count').index

    # Prepare data for box plot
    box_data = []
    box_labels = []
    for cat in top_categories:
        cat_data = df[df['Category'] == cat]['Sentiment_Subjectivity_Mean'].dropna()
        if len(cat_data) > 0:
            box_data.append(cat_data)
            box_labels.append(f"{cat}\n(n={len(cat_data)})")

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, vert=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Objective/Subjective Threshold')
    ax.set_ylabel('Subjectivity Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_title('Subjectivity Distribution for Top 15 Categories by App Count',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'subjectivity_by_category.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'subjectivity_by_category.png'}")
    plt.close()

    category_groups = []
    for cat in top_categories:
        cat_data = df[df['Category'] == cat]['Sentiment_Subjectivity_Mean'].dropna()
        if len(cat_data) >= 5:  # At least 5 samples
            category_groups.append(cat_data.values)

    if len(category_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*category_groups)
        print(f"\n Anova Test (top categories):")
        print(f"F-statistic:{f_stat:.4f}")
        print(f"P-value:{p_value:.6f}")
        if p_value < 0.05:
            print("Categories differ significantly in subjectivity (p < 0.05)")
        else:
            print("No significant difference between categories (p > 0.05)")

    return category_subjectivity


def generate_summary_report(df, corr_matrix, category_subjectivity):
    """Generate a summary report of key findings."""
    print("Summary")
    report = []
    report.append("Sentiment Analysis: Key Findings")

    # 1. Correlation findings
    report.append("1. Correlation for Installs:")
    installs_corr = corr_matrix['Installs'].drop('Installs').sort_values(ascending=False)
    for col, val in installs_corr.items():
        strength = "Strong" if abs(val) > 0.5 else "Moderate" if abs(val) > 0.3 else "Weak"
        direction = "positive" if val > 0 else "negative"
        report.append(f"  • {col}: {val:.4f} ({strength} {direction})")
    report.append("")

    # 2. Objectivity vs Subjectivity
    report.append("2. Objectivity vs Subjectivity:")
    subjectivity = df['Sentiment_Subjectivity_Mean'].dropna()
    obj_pct = (subjectivity < 0.5).sum() / len(subjectivity) * 100
    subj_pct = (subjectivity >= 0.5).sum() / len(subjectivity) * 100

    if subjectivity.mean() < 0.5:
        report.append(f"Users tend to give more objective comments")
    else:
        report.append(f"Users tend to give more subjective comments")
    report.append(f"Mean subjectivity: {subjectivity.mean():.4f}")
    report.append(f"Objective apps: {obj_pct:.1f}%")
    report.append(f"Subjective apps: {subj_pct:.1f}%")

    # 3. Category findings
    report.append("3 Subjectivity by Category:")
    most_subjective = category_subjectivity.nlargest(3, 'Mean')
    most_objective = category_subjectivity.nsmallest(3, 'Mean')

    report.append("Most Subjective Categories:")
    for cat, row in most_subjective.iterrows():
        report.append(f"{cat}: {row['Mean']:.4f} (n={int(row['Count'])})")
    report.append("")
    report.append("Most Objective Categories:")
    for cat, row in most_objective.iterrows():
        report.append(f"{cat}: {row['Mean']:.4f} (n={int(row['Count'])})")
    report.append("")

    # Print to console
    for line in report:
        print(line)

    # Save to file
    report_path = OUTPUT_DIR / 'analysis_summary.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n Saved: {report_path}")


def main():
    df = load_and_merge_data()
    # 1 Correlation analysis
    corr_matrix = analyze_correlations(df)
    # 2 Objectivity vs subjectivity
    analyze_objectivity_vs_subjectivity(df)
    # 3 Subjectivity by category
    category_subjectivity = analyze_subjectivity_by_category(df)
    # 4 Generate summary report
    generate_summary_report(df, corr_matrix, category_subjectivity)
    print(f"results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
