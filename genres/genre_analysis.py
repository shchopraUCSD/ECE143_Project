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
        'Installs': ['mean', 'median'],
        'Rating': 'mean',
        'Popularity Score': 'mean'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['Category', 'Genre', 'App_Count', 'Avg_Installs', 
                     'Median_Installs', 'Avg_Rating', 'Avg_Popularity_Score']
    
    # Add rank within category by Avg_Installs
    stats['Rank_in_Category'] = stats.groupby('Category')['Avg_Installs'].rank(
        ascending=False, method='dense'
    ).astype(int)
    
    # Sort by Category and Rank
    stats = stats.sort_values(['Category', 'Rank_in_Category'])
    
    return stats


def compute_genre_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute genre diversity metrics for each Category.
    
    Args:
        df: Original DataFrame with Genres column
        
    Returns:
        DataFrame with diversity metrics per category
    """
    df_exploded = explode_genres(df)
    
    diversity = df_exploded.groupby('Category').agg({
        'Genre': 'nunique',
        'App': 'nunique'
    }).reset_index()
    
    diversity.columns = ['Category', 'Unique_Genres', 'Total_Apps']
    diversity['Genres_per_App'] = diversity['Unique_Genres'] / diversity['Total_Apps']
    diversity = diversity.sort_values('Unique_Genres', ascending=False)
    
    return diversity


def plot_genre_diversity_by_category(
    diversity_df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres"
) -> None:
    """Plot bar chart of genre diversity by category."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(diversity_df)))
    
    plt.barh(diversity_df['Category'], diversity_df['Unique_Genres'], color=colors)
    plt.xlabel('Number of Unique Genres', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.title('Genre Diversity by Category', fontsize=14, pad=20, weight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_file = output_path / "genre_diversity_by_category.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Genre diversity plot saved to: {output_file}")


def plot_apps_per_genre_top20(
    stats_df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres"
) -> None:
    """Plot horizontal bar chart of top 20 genres by app count."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Aggregate by Genre across all categories
    genre_counts = stats_df.groupby('Genre')['App_Count'].sum().sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(genre_counts)))
    
    plt.barh(range(len(genre_counts)), genre_counts.values, color=colors)
    plt.yticks(range(len(genre_counts)), genre_counts.index)
    plt.xlabel('Number of Apps', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.title('Top 20 Genres by Number of Apps', fontsize=14, pad=20, weight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_file = output_path / "apps_per_genre_top20.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Apps per genre plot saved to: {output_file}")


def plot_genre_popularity_heatmap(
    stats_df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres",
    top_n_genres: int = 5
) -> None:
    """Plot heatmap of average installs by Category and Genre (top genres per category)."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get top N genres per category
    top_genres = stats_df[stats_df['Rank_in_Category'] <= top_n_genres].copy()
    
    # Create pivot table
    pivot = top_genres.pivot_table(
        values='Avg_Installs',
        index='Genre',
        columns='Category',
        aggfunc='mean'
    )
    
    # Use log scale for better visualization
    pivot_log = np.log10(pivot + 1)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        pivot_log,
        annot=False,
        cmap='YlOrRd',
        cbar_kws={'label': 'Log10(Average Installs + 1)'},
        linewidths=0.5
    )
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.title(f'Genre Popularity Heatmap (Top {top_n_genres} Genres per Category)', 
              fontsize=14, pad=20, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file = output_path / "genre_popularity_heatmap.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Genre popularity heatmap saved to: {output_file}")


def plot_top_genres_per_category(
    stats_df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres",
    top_n: int = 5,
    categories_to_plot: Optional[list] = None
) -> None:
    """Plot top genres for selected categories."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select categories with most genre diversity
    if categories_to_plot is None:
        category_diversity = stats_df.groupby('Category')['Genre'].nunique().sort_values(ascending=False)
        categories_to_plot = category_diversity.head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, category in enumerate(categories_to_plot):
        if idx >= len(axes):
            break
            
        cat_data = stats_df[stats_df['Category'] == category].head(top_n)
        
        ax = axes[idx]
        x_pos = np.arange(len(cat_data))
        
        # Bar chart for app count
        bars = ax.bar(x_pos, cat_data['App_Count'], alpha=0.7, color='steelblue', label='App Count')
        
        # Overlay line for average rating
        ax2 = ax.twinx()
        line = ax2.plot(x_pos, cat_data['Avg_Rating'], 'ro-', linewidth=2, markersize=8, label='Avg Rating')
        
        ax.set_xlabel('Genre', fontsize=10)
        ax.set_ylabel('Number of Apps', fontsize=10, color='steelblue')
        ax2.set_ylabel('Average Rating', fontsize=10, color='red')
        ax.set_title(f'{category}', fontsize=11, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cat_data['Genre'], rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 5)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Remove unused subplots
    for idx in range(len(categories_to_plot), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f'Top {top_n} Genres per Category (App Count & Rating)', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_path / "top_genres_per_category.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Top genres per category plot saved to: {output_file}")


def plot_multi_genre_analysis(
    df: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres"
) -> None:
    """Analyze apps with single vs multiple genres."""
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
    
    # Pie chart
    genre_type_counts = df['Genre_Type'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[0].pie(genre_type_counts.values, labels=genre_type_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[0].set_title('Distribution of Apps by Genre Count', fontsize=12, weight='bold')
    
    # Bar chart by category
    multi_genre_by_cat = df.groupby(['Category', 'Genre_Type']).size().unstack(fill_value=0)
    multi_genre_by_cat = multi_genre_by_cat.sort_values('Multiple Genres', ascending=False).head(15)
    
    multi_genre_by_cat.plot(kind='barh', stacked=True, ax=axes[1], color=colors)
    axes[1].set_xlabel('Number of Apps', fontsize=11)
    axes[1].set_ylabel('Category', fontsize=11)
    axes[1].set_title('Genre Count Distribution by Category (Top 15)', fontsize=12, weight='bold')
    axes[1].legend(title='Genre Type', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_file = output_path / "multi_genre_analysis.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Multi-genre analysis plot saved to: {output_file}")


def plot_genre_installs_boxplot(
    df_exploded: pd.DataFrame,
    root: Optional[Path] = None,
    output_dir: str = "figs/genres",
    top_n: int = 15
) -> None:
    """Plot boxplot of installs distribution for top genres."""
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get top genres by app count
    top_genres = df_exploded['Genre'].value_counts().head(top_n).index.tolist()
    df_top = df_exploded[df_exploded['Genre'].isin(top_genres)].copy()
    
    # Remove zero installs for log scale
    df_top = df_top[df_top['Installs'] > 0]
    
    # Sort by median installs
    genre_order = df_top.groupby('Genre')['Installs'].median().sort_values(ascending=False).index.tolist()
    
    plt.figure(figsize=(14, 8))
    
    # Create boxplot with color gradient
    bp = plt.boxplot(
        [df_top[df_top['Genre'] == genre]['Installs'].values for genre in genre_order],
        labels=genre_order,
        patch_artist=True,
        vert=False
    )
    
    # Color boxes by median
    medians = [df_top[df_top['Genre'] == genre]['Installs'].median() for genre in genre_order]
    norm = plt.Normalize(vmin=min(medians), vmax=max(medians))
    cmap = plt.cm.get_cmap('RdYlGn')
    
    for patch, median in zip(bp['boxes'], medians):
        patch.set_facecolor(cmap(norm(median)))
        patch.set_alpha(0.7)
    
    plt.xlabel('Installs', fontsize=12)
    plt.ylabel('Genre', fontsize=12)
    plt.title(f'Installs Distribution for Top {top_n} Genres', fontsize=14, pad=20, weight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    output_file = output_path / "genre_installs_boxplot.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Genre installs boxplot saved to: {output_file}")


def main(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    output_stats_path: str = "data_cleaning/data_processed/genre_statistics.csv"
) -> None:
    """
    Main function to run all genre analysis.
    
    Args:
        root: Root directory of the project
        csv_path: Path to cleaned apps CSV
        output_stats_path: Path to save genre statistics CSV
    """
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    print("=" * 60)
    print("GENRE ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n[1/8] Loading data...")
    csv_file = root / csv_path
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} apps")
    
    # Explode genres
    print("\n[2/8] Exploding genres...")
    df_exploded = explode_genres(df)
    print(f"After explosion: {len(df_exploded)} rows (genre-app pairs)")
    print(f"Unique genres: {df_exploded['Genre'].nunique()}")
    
    # Compute statistics
    print("\n[3/8] Computing genre statistics...")
    stats_df = compute_genre_statistics(df_exploded)
    print(f"Generated statistics for {len(stats_df)} category-genre pairs")
    
    # Save statistics
    stats_output = root / output_stats_path
    stats_df.to_csv(stats_output, index=False)
    print(f"Statistics saved to: {stats_output}")
    
    # Compute diversity
    print("\n[4/8] Computing genre diversity...")
    diversity_df = compute_genre_diversity(df)
    print(f"Top 5 categories by genre diversity:")
    print(diversity_df.head().to_string(index=False))
    
    # Generate plots
    print("\n[5/8] Generating genre diversity plot...")
    plot_genre_diversity_by_category(diversity_df, root=root)
    
    print("\n[6/8] Generating apps per genre plot...")
    plot_apps_per_genre_top20(stats_df, root=root)
    
    print("\n[7/8] Generating genre popularity heatmap...")
    plot_genre_popularity_heatmap(stats_df, root=root)
    
    print("\n[8/8] Generating additional analysis plots...")
    plot_top_genres_per_category(stats_df, root=root)
    plot_multi_genre_analysis(df, root=root)
    plot_genre_installs_boxplot(df_exploded, root=root)
    
    print("\n" + "=" * 60)
    print("GENRE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll figures saved to: {root / 'figs/genres'}")
    print(f"Statistics saved to: {stats_output}")


if __name__ == "__main__":
    main()

