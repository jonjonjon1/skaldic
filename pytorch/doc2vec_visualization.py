import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from load_data import load_data
from datetime import datetime
from typing import Literal
import random
import numpy as np
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="norse_poems.csv", help="Data filename, all data is in \"..\\data\" directory")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--doc2vec_epochs", type=int, default=20)
    parser.add_argument("--doc2vec_features", type=int, default=64)
    parser.add_argument("--doc2vec_window", default=12)
    parser.add_argument("--n_authors", type=int, default=8)
    parser.add_argument("--min_n_stanzas", type=int, default=12)
    parser.add_argument("--volume_row_fraction", type=float, default=.03)
    parser.add_argument("--dim_red", type=str, choices=["PCA", "tSNE"], default="tSNE")
    parser.add_argument("--overwrite_models", default=True)
    parser.add_argument("--save_fig", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=793)
    return parser.parse_args()

def train_doc2vec(tokenized_list: list[list[str]],
                  model_path: Path,
                  args: argparse.Namespace):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_list)]

    doc2vec_model = Doc2Vec(
        vector_size=args.doc2vec_features,
        window=args.doc2vec_window,
        min_count=5,
        # Has to be 1 to guarantee deterministic runs
        workers=1,
        epochs=args.doc2vec_epochs,
        seed=args.seed
    )

    doc2vec_model.build_vocab(tagged_data)
    doc2vec_model.train(tagged_data,
                        total_examples=doc2vec_model.corpus_count,
                        epochs=args.doc2vec_epochs)
    
    doc2vec_model.save(str(model_path))

    return doc2vec_model

def calculate_centroid_diff(vectors: np.ndarray, labels: list[str]):
    labels = np.array(labels)
    values, counts = np.unique(labels, return_counts=True)
    uniques = values[counts > 1]

    variances = {}
    for label in sorted(uniques):
        mask = (labels == label)
        grouped_vectors = vectors[mask]
        centroid = np.mean(grouped_vectors, axis=0)
        differences = grouped_vectors - centroid
        squared_dist = np.sum(np.square(differences), axis=1)
        variances[label] = np.mean(np.sqrt(squared_dist))
    return variances

def visualize_author_diff(df: pd.DataFrame,
                          vectorized_stanzas: np.ndarray,
                          n_authors: int,
                          min_stanza_freq: int,
                          dim_red: Literal["PCA", "tSNE"],
                          random_state: int,
                          plot_dir: Path|None=None) -> None:

    if dim_red == "PCA":
        pca = PCA(n_components=3, random_state=random_state)
        results = pca.fit_transform(vectorized_stanzas)
        results2d = np.array(0)
    else:
        tsne = TSNE(n_components=3,
                    perplexity=20,
                    learning_rate="auto",
                    max_iter=2000,
                    init="pca",
                    random_state=random_state)
        results = tsne.fit_transform(vectorized_stanzas)
        tsne2d = TSNE(n_components=2,
                    perplexity=20,
                    learning_rate=10,
                    max_iter=2000,
                    random_state=random_state)
        results2d = tsne2d.fit_transform(vectorized_stanzas)

    df = df[~df["author"].str.contains("Anonymous", na=False)]

    author_counts = df["author"].value_counts()
    frequent_values = author_counts[author_counts > min_stanza_freq].index.tolist()
    sampled_authors = np.random.choice(frequent_values, size=n_authors)

    logger.info(f"Sampling {len(sampled_authors)} authors from total of {len(df["author"].unique())}")
    filtered_df = df[df["author"].isin(sampled_authors)]

    def return_if_lower(x: pd.DataFrame): return x.sample(n=min(len(x), min_stanza_freq))
    df = filtered_df.groupby("author").apply(return_if_lower, include_groups=False)
    df = df.reset_index()

    if plot_dir:
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        dest_path = plot_dir / f"volumes_{dim_red}_{timestamp}"
    else: dest_path = None

    _make_plot(df["author"], results, dim_red, dest_path, results2d=results2d)

def visualize_volume_diff(df: pd.DataFrame,
                          vectorized_stanzas: np.ndarray,
                          fraction: float,
                          dim_red: Literal["PCA", "tSNE"],
                          random_state: int,
                          plot_dir: Path|None=None) -> None:

    if dim_red == "PCA":
        pca = PCA(n_components=3,  random_state=random_state)
        results = pca.fit_transform(vectorized_stanzas)
        results2d = np.array(0)
    else:
        tsne = TSNE(n_components=3,
                    perplexity=20,
                    learning_rate=10,
                    max_iter=2000,
                    init="pca",
                    random_state=random_state)
        results = tsne.fit_transform(vectorized_stanzas)
        tsne2d = TSNE(n_components=2,
                    perplexity=20,
                    learning_rate=10,
                    max_iter=2000,
                    init="pca",
                    random_state=random_state)
        results2d = tsne2d.fit_transform(vectorized_stanzas)

    chosen_volumes = df
    sampled_rows = chosen_volumes.sample(frac=fraction)
    logger.info(f"Sampling {sampled_rows.shape[0]} rows from total of {df.shape[0]}")
    df = sampled_rows

    if plot_dir:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        dest_path = plot_dir / f"volumes_{dim_red}_{timestamp}"
    else: dest_path = None

    _make_plot(df["volume"], results, dim_red, dest_path, results2d=results2d)

def _make_plot(series: pd.Series,
               results: np.ndarray,
               dim_red: Literal["PCA", "tSNE"],
               dest_path: Path,
               results2d: np.ndarray|None=None) -> None:
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    labels = series.unique()

    colors = sns.color_palette("hls", len(labels))
    label_color_map = dict(zip(labels, colors))

    for label in sorted(labels):
        indices = [i for i, l in enumerate(series) if l == label]
        
        ax.scatter(
            results[indices,0],
            results[indices,1],
            results[indices,2],
            label=label,
            color=label_color_map[label],
            s=60,
            alpha=.7,
            edgecolors="w"
        )

    ax.set_title(f"{dim_red} embeddings of documents (stanzas)", fontsize=15)
    ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f"{dim_red}1")
    ax.set_ylabel(f"{dim_red}2")
    ax.set_zlabel(f"{dim_red}3")

    plt.tight_layout()
    if dest_path:
        plt.savefig(f"{dest_path}_3D.png")
    else:
        plt.show()

    fig, ax = plt.subplots(figsize=(9, 8))

    for label in sorted(labels):
        indices = [i for i, l in enumerate(series) if l == label]
        
        if not results2d.any():
            ax.scatter(
                results[indices,0],
                results[indices,1],
                label=label,
                color=label_color_map[label],
                s=60,
                alpha=.7,
                edgecolors="w"
            )
        else:
            ax.scatter(
                results2d[indices,0],
                results2d[indices,1],
                label=label,
                color=label_color_map[label],
                s=60,
                alpha=.7,
                edgecolors="w"
            )

    ax.grid(alpha=.5)
    ax.set_title(f"{dim_red} embeddings of documents (stanzas)", fontsize=15)
    ax.legend(title="Labels")
    ax.set_xlabel(f"{dim_red}1")
    ax.set_ylabel(f"{dim_red}2")

    if dest_path:
        plt.savefig(f"{dest_path}_2D.png")
    else:
        plt.show()

if __name__=="__main__":
    args = parse_args()
    seed_everything(args.seed)
    current_dir = Path(__file__).parent

    df = load_data(args)

    bin_dir = current_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    doc2vec_default = bin_dir / "doc2vec.model"
    if not args.overwrite_models and next(bin_dir.glob("doc2vec*"), None):
        doc2vec_model = Doc2Vec.load(str(doc2vec_default))
    else:
        doc2vec_model = train_doc2vec(df["tokenized"], doc2vec_default, args)

    tokenized = df["tokenized"]
    stanza_vectors = np.array([doc2vec_model.dv[i] for i in range(len(tokenized))])
    vectors_normalized = normalize(stanza_vectors, axis=1, norm="l2")
    
    label = "author"
    author_silhouette = silhouette_score(vectors_normalized, df[label].copy().tolist(), metric="cosine")
    author_variances = calculate_centroid_diff(vectors_normalized, df[label].copy().tolist())
    author_mean_variance = np.array(list(author_variances.values())).mean()
    logger.info(f"Author clusters\tSilhouette score: {author_silhouette}\tMean variance: {author_mean_variance}")
    label = "volume"
    volume_silhouette = silhouette_score(vectors_normalized, df[label].copy().tolist(), metric="cosine")
    volume_variances = calculate_centroid_diff(vectors_normalized, df[label].copy().tolist())
    volume_mean_variance = np.array(list(volume_variances.values())).mean()
    logger.info(f"Volume clusters\tSilhouette score: {volume_silhouette}\tMean variance: {volume_mean_variance}")


    if args.save_fig:
        plot_dir = current_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        plot_dir = None

    visualize_author_diff(df,
                          stanza_vectors,
                          args.n_authors,
                          args.min_n_stanzas,
                          args.dim_red,
                          args.seed,
                          plot_dir=plot_dir)
    
    visualize_volume_diff(df,
                          stanza_vectors,
                          args.volume_row_fraction,
                          args.dim_red,
                          args.seed,
                          plot_dir=plot_dir)