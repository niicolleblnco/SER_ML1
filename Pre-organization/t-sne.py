import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from load import build_df_from_ravdess
from features import SERAudioDataset
from sklearn.manifold import TSNE

DATA_ROOT = "data"
N_MFCC = 20
SAMPLE_RATE = 16000

def extract_features(df):
    ds = SERAudioDataset(
        df,
        sample_rate=SAMPLE_RATE,
        feature_type="mfcc",
        n_mfcc=N_MFCC
    )

    feats = []
    labs = []
    unique = df["Emotion"].unique()
    mapping = {v: i for i, v in enumerate(unique)}

    for i in range(len(ds)):
        x, _, _ = ds[i]
        label = df.iloc[i]["Emotion"]

        arr = x.cpu().numpy()
        
        if arr.ndim != 2:
            continue

        mean_vec = arr.mean(axis=1)
        feats.append(mean_vec)
        labs.append(mapping[label])

    feats = np.array(feats)
    labs = np.array(labs)

    print("valid samples:", feats.shape[0])
    return feats, labs

def run_pca():
    df = build_df_from_ravdess(DATA_ROOT)
    feats, labs = extract_features(df)

    print("Feature shape:", feats.shape)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(feats)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=labs, s=10, cmap="tab10")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection of MFCC features")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()

def run_tsne():
    df = build_df_from_ravdess(DATA_ROOT)
    feats, labs = extract_features(df)

    print("Feature shape:", feats.shape)

    tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1500,
    metric="euclidean",
    init="pca"
    )

    proj = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(proj[:, 0], proj[:, 1], c=labs, s=10, cmap="tab10")
    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.title("t-SNE projection of MFCC features")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tsne()