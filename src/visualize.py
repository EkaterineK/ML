# for PCA visualization

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_pca(X, y, save_path="../results/pca_visualization.png"):
    """
    Reduces features to 2D using PCA and plots points colored by diabetes outcome.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=X_pca[:,0], 
        y=X_pca[:,1], 
        hue=y, 
        palette="coolwarm", 
        alpha=0.7
    )

    plt.title("PCA Visualization of Diabetes Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(save_path)  
    plt.close()

