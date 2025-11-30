from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler


def categorize_installs(installs):
    """
    Categorize installs into 5 levels.
    
    Args:
        installs: Number of installs
        
    Returns:
        int: Install level (0-4)
    """
    if installs < 10000:
        return 0  # Low
    elif installs < 100000:
        return 1  # Medium-Low
    elif installs < 1000000:
        return 2  # Medium
    elif installs < 10000000:
        return 3  # Medium-High
    else:
        return 4  # High


def compute_rating_dynamic_score(row):
    """
    Compute dynamic rating score based on app age.
    New apps need higher ratings, old apps can have slightly lower ratings.
    
    Args:
        row: DataFrame row with Rating and Last Updated
        
    Returns:
        float: Rating score (0-1)
    """
    rating = row['Rating']
    
    if pd.isna(rating):
        return 0.5  
    
    try:
        last_updated = pd.to_datetime(row['Last Updated'])
        days_since_update = (pd.Timestamp.now() - last_updated).days
    except:
        days_since_update = 365  
    
    # New app (< 6 months)
    if days_since_update < 180:
        if rating >= 4.5:
            return 1.0
        elif rating >= 4.0:
            return 0.7
        else:
            return 0.3
        
    # Old app (>= 6 months)
    else:
        if rating >= 4.2:
            return 1.0
        elif rating >= 3.8:
            return 0.6
        else:
            return 0.2


def compute_developer_activity_score(row):
    """
    Compute developer activity score combining update recency and app age.
    Ideal: Recently updated + Old Android version = Long-term maintained mature app.
    
    Args:
        row: DataFrame row with Last Updated and Android Ver
        
    Returns:
        float: Developer activity score (0-1)
    """
    try:
        last_updated = pd.to_datetime(row['Last Updated'])
        days_since_update = (pd.Timestamp.now() - last_updated).days
    except:
        days_since_update = 365  
    
    # Update recency score (more recent = better)
    if days_since_update < 90:
        update_score = 1.0
    elif days_since_update < 180:
        update_score = 0.8
    elif days_since_update < 365:
        update_score = 0.6
    else:
        update_score = 0.3
    
    android_ver = row['Android Ver']
    if pd.isna(android_ver):
        version_score = 0.5 
    elif android_ver <= 3:
        version_score = 1.0  
    elif android_ver == 4:
        version_score = 0.7 
    else:
        version_score = 0.5  
    
    activity_score = 0.4 * update_score + 0.6 * version_score
    
    return activity_score


def compute_content_rating_targeted_score(content_rating):
    """
    Compute content rating targeting score.
    More targeted ratings (Teen, Mature, 10+) are better than generic (Everyone).
    
    Args:
        content_rating: Content rating string
        
    Returns:
        float: Targeting score (0-1)
    """
    if pd.isna(content_rating):
        return 0.5
    
    content_rating = str(content_rating).strip()
    
    if content_rating in ['Teen', 'Mature 17+', 'Everyone 10+']:
        return 1.0  
    elif content_rating == 'Everyone':
        return 0.5  
    else:
        return 0.3  


def get_genre_focus_score(genres_str, genre_scores_dict):
    """
    Get genre focus score from pre-computed genre recommendations.
    
    Args:
        genres_str: Semicolon-separated genres string
        genre_scores_dict: Dictionary mapping genre to focus score
        
    Returns:
        float: Genre focus score (0-1)
    """
    if pd.isna(genres_str):
        return 0.5 
    

    first_genre = str(genres_str).split(';')[0].strip()
    return genre_scores_dict.get(first_genre, 0.5)


def engineer_features(df, genre_scores_dict=None):
    """
    Engineer all features for prediction.
    
    Args:
        df: Input DataFrame
        genre_scores_dict: Optional genre scores dictionary
        
    Returns:
        pd.DataFrame: Feature matrix
    """
    X = pd.DataFrame(index=df.index)
    
    reviews_log = np.log1p(df['Reviews'].fillna(0))
    X['reviews_log_1'] = reviews_log
    X['reviews_log_2'] = reviews_log
    X['reviews_log_3'] = reviews_log
    X['reviews_log_4'] = reviews_log
    X['reviews_log_5'] = reviews_log
    
    X['rating_dynamic'] = df.apply(compute_rating_dynamic_score, axis=1)
    

    X['price_free'] = (df['Price'].fillna(0) == 0).astype(int)
    

    X['dev_activity'] = df.apply(compute_developer_activity_score, axis=1)
    
    X['content_targeted'] = df['Content Rating'].apply(compute_content_rating_targeted_score)
    

    if genre_scores_dict is not None:
        X['genre_focus'] = df['Genres'].apply(
            lambda x: get_genre_focus_score(x, genre_scores_dict)
        )
    
    category_dummies = pd.get_dummies(df['Category'], prefix='Cat', drop_first=True)
    X = pd.concat([X, category_dummies], axis=1)
    
    return X


def train_logistic_regression(
    root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    genre_scores_path: str = "data_cleaning/data_processed/genre_focus_recommendations.csv",
    output_dir: str = "figs/prediction"
):
    """
    Train Logistic Regression model for install level prediction.
    
    Args:
        root: Root directory
        csv_path: Path to cleaned data
        genre_scores_path: Path to genre scores
        output_dir: Output directory for figures
    """
    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("LOGISTIC REGRESSION MODEL FOR INSTALL PREDICTION")
    print("=" * 70)
    

    print("\n[1/8] Loading data...")
    df = pd.read_csv(root / csv_path)
    print(f"Loaded {len(df)} apps")

    df = df[df['Installs'].notna()].copy()
    print(f"After removing missing Installs: {len(df)} apps")
    

    print("\n[2/8] Loading genre focus scores...")
    try:
        genre_scores = pd.read_csv(root / genre_scores_path)
        genre_scores_dict = dict(zip(genre_scores['Genre'], genre_scores['Focus_Score']))
        print(f"Loaded {len(genre_scores_dict)} genre scores")
    except:
        print("Genre scores not found, skipping this feature")
        genre_scores_dict = None
    

    print("\n[3/8] Creating target variable...")
    y = df['Installs'].apply(categorize_installs)
    
    level_names = ['Low (<10K)', 'Medium-Low (10K-100K)', 'Medium (100K-1M)', 
                   'Medium-High (1M-10M)', 'High (>10M)']
    
    print("Target distribution:")
    for level in range(5):
        count = (y == level).sum()
        pct = count / len(y) * 100
        print(f"  Level {level} ({level_names[level]}): {count} ({pct:.1f}%)")
    

    print("\n[4/8] Engineering features...")
    X = engineer_features(df, genre_scores_dict)
    print(f"Created {X.shape[1]} features")
    print(f"Feature names: {X.columns.tolist()[:10]}... (showing first 10)")
    

    print("\n[5/8] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\n[6/8] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n[7/8] Training Logistic Regression...")
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("Training complete!")
    
    # 8. Evaluate
    print("\n[8/8] Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=level_names, zero_division=0))
    
    print("\n[9/9] Generating visualizations...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    short_names = ['L0\n(<10K)',
                'L1\n(10K–100K)',
                'L2\n(100K–1M)',
                'L3\n(1M–10M)',
                'L4\n(>10M)']
    plt.figure(figsize=(7, 6)) 
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=short_names,
        yticklabels=short_names,
        square=True,            
        linewidths=0.5,          
        linecolor='white',
        cbar_kws={"shrink": 0.8}, 
        annot_kws={"size": 11}   
    )

    ax.set_xlabel('Predicted Level', fontsize=12, weight='bold')
    ax.set_ylabel('Actual Level', fontsize=12, weight='bold')
    ax.set_title('Confusion Matrix – Logistic Regression',
                fontsize=14, weight='bold', pad=12)

    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix_logistic.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    
    coef_high = model.coef_[-1]

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': coef_high
    })

    mask_reviews = feature_importance['feature'].str.startswith('reviews_log_')
    first_review_idx = feature_importance[mask_reviews].index.min()
    feature_importance = feature_importance.drop(
        feature_importance[mask_reviews & (feature_importance.index != first_review_idx)].index
    )


    feature_importance.loc[first_review_idx, 'feature'] = 'Review'


    feature_importance = feature_importance.sort_values('coefficient', ascending=False)

    top_features = pd.concat([
        feature_importance.head(10),
        feature_importance.tail(5)
    ])



    feature_importance = feature_importance.sort_values('coefficient', ascending=False)
    top_features = pd.concat([
        feature_importance.head(10),
        feature_importance.tail(5)
    ])

    def pretty_name(feat: str) -> str:
        if feat == 'rating_dynamic':
            return 'Dynamic rating score'
        if feat == 'price_free':
            return 'Free app (Price = 0)'
        if feat == 'dev_activity':
            return 'Developer activity score'
        if feat == 'content_targeted':
            return 'Content rating targeting score'
        if feat == 'genre_focus':
            return 'Genre focus score'
        if feat == 'Review count (log)':
            return 'Review count (log)'


        if feat.startswith('Cat_'):
            cat = feat[len('Cat_'):].replace('_', ' ')
            return f'Category: {cat.title()}'


        return feat.replace('_', ' ').title()

    top_features['pretty_feature'] = top_features['feature'].map(pretty_name)
    
    plt.figure(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
    plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)

    plt.yticks(range(len(top_features)), top_features['pretty_feature'])
    plt.xlabel('Coefficient Value', fontsize=12, weight='bold')
    plt.ylabel('Feature', fontsize=12, weight='bold')
    plt.title('Top Features for Predicting High Installs (Level 4)', 
            fontsize=14, weight='bold', pad=20)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path / "feature_coefficients_logistic.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path / 'feature_coefficients_logistic.png'}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(y_test, bins=5, range=(-0.5, 4.5), alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Install Level', fontsize=11, weight='bold')
    axes[0].set_ylabel('Count', fontsize=11, weight='bold')
    axes[0].set_title('Actual Distribution (Test Set)', fontsize=12, weight='bold')
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['L0', 'L1', 'L2', 'L3', 'L4'])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].hist(y_pred, bins=5, range=(-0.5, 4.5), alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('Install Level', fontsize=11, weight='bold')
    axes[1].set_ylabel('Count', fontsize=11, weight='bold')
    axes[1].set_title('Predicted Distribution (Test Set)', fontsize=12, weight='bold')
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(['L0', 'L1', 'L2', 'L3', 'L4'])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / "prediction_distribution_logistic.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'prediction_distribution_logistic.png'}")
    
    correct = (y_test == y_pred)
    reviews_test = X_test['reviews_log_1'].values
    
    plt.figure(figsize=(12, 6))
    plt.scatter(reviews_test[correct], y_test[correct], 
                c='green', alpha=0.5, s=20, label='Correct Prediction')
    plt.scatter(reviews_test[~correct], y_test[~correct], 
                c='red', alpha=0.5, s=20, label='Wrong Prediction')
    plt.xlabel('log(Reviews + 1)', fontsize=12, weight='bold')
    plt.ylabel('Actual Install Level', fontsize=12, weight='bold')
    plt.title('Reviews vs Install Level (Colored by Prediction Correctness)', 
              fontsize=14, weight='bold', pad=20)
    plt.yticks(range(5), ['L0', 'L1', 'L2', 'L3', 'L4'])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "reviews_vs_installs_correctness.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'reviews_vs_installs_correctness.png'}")
    

    print("\n[10/10] Saving detailed results...")
    
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[1]
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_path.parent / 'prediction' / 'logistic_results.csv', index=False)
    
    feature_importance.to_csv(output_path.parent / 'prediction' / 'feature_importance_logistic.csv', index=False)

    cm_df = pd.DataFrame(cm, index=level_names, columns=level_names)
    cm_df.to_csv(output_path.parent / 'prediction' / 'confusion_matrix_logistic.csv')
    
    print(f"\nResults saved to: {output_path.parent / 'prediction'}")
    
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION COMPLETE")
    print("=" * 70)
    
    return model, scaler, X, y, X_test, y_test, y_pred, feature_importance


if __name__ == "__main__":
    model, scaler, X, y, X_test, y_test, y_pred, feature_importance = train_logistic_regression()
    
    print("\n" + "=" * 70)
    print("TOP 10 MOST IMPORTANT FEATURES (for High Installs):")
    print("=" * 70)
    print(feature_importance.head(10).to_string(index=False))

