import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_anomalies(X, results):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, anomalies) in enumerate(results.items(), 1):
        plt.subplot(1, 3, i)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomalies, cmap='coolwarm', s=8, alpha=0.6)
        plt.title(f"{name}\n(red = anomaly)")
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
    
    plt.tight_layout()
    plt.savefig('anomalies_pca.png')
    plt.close()
    print("Visualization saved: anomalies_pca.png")