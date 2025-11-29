
from pathlib import Path
from typing import  Optional
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
# from apps_cleaning import head_with_csv_lines, clean_googleplay_apps, apps_basic_stats

def create_viz_for_content_rating(root: Optional[Path] = None,
    csv_path: str = "data_cleaning/data_processed/googleplaystore_clean.csv",
    out_dir: str = "figs/content_rating"):

    if root is None:
        root = Path(__file__).parent.parent
    else:
        root = Path(root)
    
    csv_file = root / csv_path
    out_dir = root / out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    # What is the count of apps under each content rating?
    plt.figure(figsize=(20,6))
    counts = df['Content Rating'].value_counts()
    sizes = counts.values
    plt.scatter(counts.index, counts.values, s=sizes, alpha=0.6,c=counts.values,cmap="viridis")
    plt.xlabel("Category Rating")
    plt.ylabel("Count")
    plt.colorbar()
    plt.title("Count of Content Rating of App")
    plt.savefig(out_dir / f"Content_Rating_vs_Count.png", bbox_inches="tight")
    plt.close()
    
    
    #What is the content rating plot for each category? Which category has which kind of content rating?
    plt.figure(figsize=(20,6))
    categories=df["Category"].unique()
    palette = sns.color_palette("Set2", n_colors=len(categories))
    grid=sns.FacetGrid(df,col="Content Rating",col_wrap=3,height=7,sharey=False)
    grid.map_dataframe(sns.countplot,x="Category",palette=palette)
    for axis in grid.axes.flatten():
        axis.set_xticklabels([])
    patches = [mpatches.Patch(color=palette[i], label=cat) for i, cat in enumerate(categories)]
    plt.legend(handles=patches, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')

    grid.set_ylabels("Category Count")  
    grid.figure.suptitle(f"Category Count  per Content Rating Plot")
    plt.tight_layout()
    plt.savefig(out_dir / f"Content_Rating_vs_Category_Count.png", bbox_inches="tight")
    plt.close()
 
    
    #Does the Content Rating Affect the number of installs? That is if it is tagged to wider section of audience will that affect the number of installs?
    plt.figure(figsize=(12,6))
    norm = mcolors.Normalize(vmin=0, vmax=5)
    colors = [cm.plasma(norm(i)) for i in range(6)]
    sns.boxplot(data=df, x="Content Rating", y="Installs", palette=colors,fliersize=3,boxprops={'alpha':0.8})
    plt.yscale('log')
    plt.xlabel("Content Rating")
    plt.ylabel("Installs")
    plt.xticks(rotation=45)
    plt.title(f"Content Rating  vs Installs Plot")
    plt.savefig(out_dir / f"Content_Rating_vs_Installs.png", bbox_inches="tight")
    plt.close()
 
    

    #What are the best installs values for top 2
    for m in counts.index:
        df1=df[df["Content Rating"] == m]
        info_graphs= df1['Category'].value_counts().nlargest(2).index
        df1=df1[df1["Category"].isin(info_graphs)]
        plt.figure(figsize=(5,4))
        sns.boxplot(data=df1, x="Category", y="Installs", palette="pastel",linewidth=1.1,fliersize=3)
        plt.yscale('log')
        plt.title(f"Top 2 Categories for {m}")
        plt.xlabel("Category")
        plt.ylabel("Installs")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.savefig(out_dir / f"Top 2 Categories for {m}.png", bbox_inches="tight")
        plt.close()

    


if __name__ == "__main__":
    create_viz_for_content_rating()












    

