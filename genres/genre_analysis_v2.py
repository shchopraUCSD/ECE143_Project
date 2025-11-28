from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the Genres column to create one row per genre per app.
    
    Args:
        df: DataFrame with Genres column containing semicolon-separated values
        
    Returns:
        DataFrame with exploded genres, one row per genre per app
    """
    # Split genres by semicolon and explode
    df_exploded = df.copy()
    df_exploded['Genre'] = df_exploded['Genres'].str.split(';')
    df_exploded = df_exploded.explode('Genre')
    
    # Clean up genre names (strip whitespace, standardize)
    df_exploded['Genre'] = df_exploded['Genre'].str.strip()
    
    # Remove rows with missing genres
    df_exploded = df_exploded[df_exploded['Genre'].notna() & (df_exploded['Genre'] != '')]
    
    return df_exploded


def compute_genre_statistics(df_exploded: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each Genre within each Category.
    
    Args:
        df_exploded: DataFrame with exploded genres
        
    Returns:
        DataFrame with genre statistics
    """
    # Group by Category and Genre
    stats = df_exploded.groupby(['Category', 'Genre']).agg({
        'App': 'nunique',  # Count unique apps
        'Installs': ['mean', 'median', 'sum'],
        'Rating': 'mean',
        'Popularity Score': 'mean'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['Category', 'Genre', 'App_Count', 'Avg_Installs', 
                     'Median_Installs', 'Total_Installs', 'Avg_Rating', 'Avg_Popularity_Score']
    
    # Add rank within category by Avg_Installs
    stats['Rank_in_Category'] = stats.groupby('Category')['Avg_Installs'].rank(
        ascending=False, method='dense'
    ).astype(int)
    
    # Sort by Category and Rank
    stats = stats.sort_values(['Category', 'Rank_in_Category'])
    
    return stats


def plot_family_game_genres_comparison(
    stats_df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres"
) -> None:
    """
    Plot separate bar charts for FAMILY and GAME categories showing app count and installs.
    """
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot for FAMILY category
    family_data = stats_df[stats_df['Category'] == 'FAMILY'].sort_values('Total_Installs', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(family_data))
    width = 0.35
    
    # Normalize installs for better visualization (divide by 1M)
    installs_normalized = family_data['Total_Installs'] / 1_000_000
    
    bars1 = ax.bar(x - width/2, family_data['App_Count'], width, label='Number of Apps', color='blue', alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, installs_normalized, width, label='Total Installs (M)', color='red', alpha=0.7)
    
    ax.set_xlabel('Genre', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Apps', fontsize=12, weight='bold', color='blue')
    ax2.set_ylabel('Total Installs (Millions)', fontsize=12, weight='bold', color='red')
    ax.set_title('FAMILY Category: Genres by App Count and Total Installs', fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(family_data['Genre'], rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_file = output_path / "family_genres_apps_installs.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"FAMILY genre comparison saved to: {output_file}")
    
    # Plot for GAME category
    game_data = stats_df[stats_df['Category'] == 'GAME'].sort_values('Total_Installs', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(game_data))
    
    # Normalize installs for better visualization (divide by 1M)
    installs_normalized = game_data['Total_Installs'] / 1_000_000
    
    bars1 = ax.bar(x - width/2, game_data['App_Count'], width, label='Number of Apps', color='blue', alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, installs_normalized, width, label='Total Installs (M)', color='red', alpha=0.7)
    
    ax.set_xlabel('Genre', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Apps', fontsize=12, weight='bold', color='blue')
    ax2.set_ylabel('Total Installs (Millions)', fontsize=12, weight='bold', color='red')
    ax.set_title('GAME Category: Genres by App Count and Total Installs', fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(game_data['Genre'], rotation=45, ha='right')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_file = output_path / "game_genres_apps_installs.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"GAME genre comparison saved to: {output_file}")


def plot_multi_genre_analysis_fixed(
    df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres"
) -> None:
    """Analyze apps with single vs multiple genres - with consistent colors."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Count genres per app
    df['Genre_Count'] = df['Genres'].str.count(';') + 1
    df.loc[df['Genres'].isna(), 'Genre_Count'] = 0
    
    # Categorize
    df['Genre_Type'] = df['Genre_Count'].apply(
        lambda x: 'No Genre' if x == 0 else ('Single Genre' if x == 1 else 'Multiple Genres')
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define consistent colors
    color_map = {
        'No Genre': '#ff9999',
        'Single Genre': '#66b3ff', 
        'Multiple Genres': '#99ff99'
    }
    
    # Pie chart
    genre_type_counts = df['Genre_Type'].value_counts()
    colors_pie = [color_map[label] for label in genre_type_counts.index]
    axes[0].pie(genre_type_counts.values, labels=genre_type_counts.index, autopct='%1.1f%%',
                colors=colors_pie, startangle=90)
    axes[0].set_title('Distribution of Apps by Genre Count', fontsize=12, weight='bold')
    
    # Bar chart by category
    multi_genre_by_cat = df.groupby(['Category', 'Genre_Type']).size().unstack(fill_value=0)
    multi_genre_by_cat = multi_genre_by_cat.sort_values('Multiple Genres', ascending=False).head(15)
    
    # Reorder columns to match color_map
    column_order = ['No Genre', 'Single Genre', 'Multiple Genres']
    column_order = [col for col in column_order if col in multi_genre_by_cat.columns]
    multi_genre_by_cat = multi_genre_by_cat[column_order]
    colors_bar = [color_map[col] for col in column_order]
    
    multi_genre_by_cat.plot(kind='barh', stacked=True, ax=axes[1], color=colors_bar)
    axes[1].set_xlabel('Number of Apps', fontsize=11)
    axes[1].set_ylabel('Category', fontsize=11)
    axes[1].set_title('Genre Count Distribution by Category (Top 15)', fontsize=12, weight='bold')
    axes[1].legend(title='Genre Type', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_file = output_path / "multi_genre_analysis_fixed.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Multi-genre analysis (fixed colors) saved to: {output_file}")


def plot_all_games_genre_installs(
    df_exploded: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres"
) -> None:
    """
    Plot installs distribution for all GAME genres.
    """
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter for GAME category
    game_data = df_exploded[df_exploded['Category'] == 'GAME'].copy()
    
    # Aggregate by Genre
    genre_stats = game_data.groupby('Genre').agg({
        'App': 'nunique',
        'Installs': ['sum', 'mean', 'median']
    }).reset_index()
    
    genre_stats.columns = ['Genre', 'App_Count', 'Total_Installs', 'Avg_Installs', 'Median_Installs']
    genre_stats = genre_stats.sort_values('Total_Installs', ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left plot: Total Installs by Genre
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(genre_stats)))
    ax1.barh(genre_stats['Genre'], genre_stats['Total_Installs'] / 1_000_000, color=colors1)
    ax1.set_xlabel('Total Installs (Millions)', fontsize=12, weight='bold')
    ax1.set_ylabel('Genre', fontsize=12, weight='bold')
    ax1.set_title('GAME Category: Total Installs by Genre', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right plot: App Count vs Average Installs
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(genre_stats)))
    scatter = ax2.scatter(genre_stats['App_Count'], genre_stats['Avg_Installs'] / 1_000_000, 
                         s=200, c=range(len(genre_stats)), cmap='plasma', alpha=0.6, edgecolors='black')
    
    # Add genre labels
    for idx, row in genre_stats.iterrows():
        ax2.annotate(row['Genre'], (row['App_Count'], row['Avg_Installs'] / 1_000_000),
                    fontsize=8, ha='right', va='bottom')
    
    ax2.set_xlabel('Number of Apps', fontsize=12, weight='bold')
    ax2.set_ylabel('Average Installs (Millions)', fontsize=12, weight='bold')
    ax2.set_title('GAME Category: App Count vs Average Installs', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_path / "game_genres_installs_analysis.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"GAME genres installs analysis saved to: {output_file}")


def analyze_genre_focus_recommendations(
    stats_df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres"
) -> pd.DataFrame:
    """
    Analyze which genres should be focused on based on app count and installs.
    Returns a DataFrame with recommendations.
    """
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics
    genre_summary = stats_df.groupby('Genre').agg({
        'App_Count': 'sum',
        'Total_Installs': 'sum',
        'Avg_Installs': 'mean'
    }).reset_index()
    
    # Calculate efficiency: Avg Installs per App
    genre_summary['Installs_per_App'] = genre_summary['Total_Installs'] / genre_summary['App_Count']
    
    # Normalize metrics for scoring
    genre_summary['App_Count_Norm'] = (genre_summary['App_Count'] - genre_summary['App_Count'].min()) / \
                                       (genre_summary['App_Count'].max() - genre_summary['App_Count'].min())
    genre_summary['Installs_Norm'] = (genre_summary['Total_Installs'] - genre_summary['Total_Installs'].min()) / \
                                      (genre_summary['Total_Installs'].max() - genre_summary['Total_Installs'].min())
    genre_summary['Efficiency_Norm'] = (genre_summary['Installs_per_App'] - genre_summary['Installs_per_App'].min()) / \
                                        (genre_summary['Installs_per_App'].max() - genre_summary['Installs_per_App'].min())
    
    # Combined score: 30% app count, 40% total installs, 30% efficiency
    genre_summary['Focus_Score'] = (0.3 * genre_summary['App_Count_Norm'] + 
                                    0.4 * genre_summary['Installs_Norm'] + 
                                    0.3 * genre_summary['Efficiency_Norm'])
    
    genre_summary = genre_summary.sort_values('Focus_Score', ascending=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Top 15 genres by Focus Score
    top_genres = genre_summary.head(15)
    
    # Plot 1: Focus Score
    colors = plt.cm.RdYlGn(top_genres['Focus_Score'])
    axes[0, 0].barh(top_genres['Genre'], top_genres['Focus_Score'], color=colors)
    axes[0, 0].set_xlabel('Focus Score', fontsize=11, weight='bold')
    axes[0, 0].set_ylabel('Genre', fontsize=11, weight='bold')
    axes[0, 0].set_title('Top 15 Genres by Focus Score', fontsize=12, weight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: App Count
    axes[0, 1].barh(top_genres['Genre'], top_genres['App_Count'], color='#4472C4', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Apps', fontsize=11, weight='bold')
    axes[0, 1].set_ylabel('Genre', fontsize=11, weight='bold')
    axes[0, 1].set_title('App Count for Top Genres', fontsize=12, weight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Total Installs
    axes[1, 0].barh(top_genres['Genre'], top_genres['Total_Installs'] / 1_000_000, color='#ED7D31', alpha=0.7)
    axes[1, 0].set_xlabel('Total Installs (Millions)', fontsize=11, weight='bold')
    axes[1, 0].set_ylabel('Genre', fontsize=11, weight='bold')
    axes[1, 0].set_title('Total Installs for Top Genres', fontsize=12, weight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Installs per App (Efficiency)
    axes[1, 1].barh(top_genres['Genre'], top_genres['Installs_per_App'] / 1_000_000, color='#70AD47', alpha=0.7)
    axes[1, 1].set_xlabel('Avg Installs per App (Millions)', fontsize=11, weight='bold')
    axes[1, 1].set_ylabel('Genre', fontsize=11, weight='bold')
    axes[1, 1].set_title('Efficiency: Installs per App', fontsize=12, weight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Genre Focus Recommendations Analysis', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_path / "genre_focus_recommendations.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Genre focus recommendations saved to: {output_file}")
    
    # Save recommendations to CSV
    recommendations_file = root / "data_cleaning/data_processed/genre_focus_recommendations.csv"
    genre_summary[['Genre', 'App_Count', 'Total_Installs', 'Installs_per_App', 'Focus_Score']].to_csv(
        recommendations_file, index=False
    )
    print(f"Recommendations CSV saved to: {recommendations_file}")
    
    return genre_summary


def main(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_stats_path: str = "data_cleaning/data_processed/genre_statistics_v2.csv"
) -> None:
    """
    Main function to run improved genre analysis.
    """
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    print("=" * 60)
    print("IMPROVED GENRE ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1/6] Loading data...")
    csv_file = root / csv_path
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} apps")
    
    # Explode genres
    print("\n[2/6] Exploding genres...")
    df_exploded = explode_genres(df)
    print(f"After explosion: {len(df_exploded)} rows (genre-app pairs)")
    
    # Compute statistics
    print("\n[3/6] Computing genre statistics...")
    stats_df = compute_genre_statistics(df_exploded)
    print(f"Generated statistics for {len(stats_df)} category-genre pairs")
    
    # Save statistics
    stats_output = root / output_stats_path
    stats_df.to_csv(stats_output, index=False)
    print(f"Statistics saved to: {stats_output}")
    
    # Generate improved plots
    print("\n[4/6] Generating FAMILY and GAME genre comparisons...")
    plot_family_game_genres_comparison(stats_df, root=root)
    
    print("\n[5/6] Generating multi-genre analysis with fixed colors...")
    plot_multi_genre_analysis_fixed(df, root=root)
    
    print("\n[6/6] Generating GAME genres installs analysis...")
    plot_all_games_genre_installs(df_exploded, root=root)
    
    print("\n[7/7] Generating genre focus recommendations...")
    recommendations = analyze_genre_focus_recommendations(stats_df, root=root)
    
    print("\n" + "=" * 60)
    print("IMPROVED GENRE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll figures saved to: {root / 'figs/genres'}")
    print(f"Statistics saved to: {stats_output}")
    print("\nTop 5 Recommended Genres to Focus On:")
    print(recommendations[['Genre', 'App_Count', 'Total_Installs', 'Focus_Score']].head().to_string(index=False))


if __name__ == "__main__":
    main()

