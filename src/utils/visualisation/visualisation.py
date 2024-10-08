import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for anchor, _, _, batch_labels in tqdm(dataloader, desc="Generating embeddings"):
            images = anchor.to(device)
            outputs = model.forward_once(images)
            embeddings.append(outputs.cpu().numpy())
            labels.extend(batch_labels.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels

def visualize_embeddings(embeddings, labels, title, save_path=None):
    # Reduce number of samples if too large
    max_samples = 500
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    print(f"Performing t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter, ticks=range(5))  # Assuming 5 classes
    plt.title(title)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_embeddings_umap(embeddings, labels, title):
    n_samples = min(len(embeddings), len(labels))
    embeddings = embeddings[:n_samples]
    labels = labels[:n_samples]

    print("Performing PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    
    # Add variance explained
    var_explained = pca.explained_variance_ratio_
    plt.xlabel(f'PCA component 1 ({var_explained[0]:.2%} variance explained)')
    plt.ylabel(f'PCA component 2 ({var_explained[1]:.2%} variance explained)')
    
    plt.show()