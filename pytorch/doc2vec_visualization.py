import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from load_data import load_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Input files and, ratio of input used and merges performed")
    parser.add_argument("--data_file", type=str, default="norse_poems.csv", help="Data filename, all data is in \"..\\data\" directory")
    parser.add_argument("--vocab_size", default=1000)
    parser.add_argument("--doc2vec_epochs", default=50)
    parser.add_argument("--doc2vec_features", default=32)
    parser.add_argument("--doc2vec_window", default=5)
    parser.add_argument("--overwrite_models", default=False)
    parser.add_argument("--n_authors", default=5)
    parser.add_argument("--volume_row_fraction", default=.1)
    return parser.parse_args()

def train_doc2vec(tokenized_list: list[list[str]],
                  model_path: Path,
                  args: argparse.Namespace):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_list)]

    doc2vec_model = Doc2Vec(
        vector_size=args.doc2vec_features,
        window=args.doc2vec_window,
        min_count=1,
        workers=4,
        epochs=args.doc2vec_epochs
    )

    doc2vec_model.build_vocab(tagged_data)
    doc2vec_model.train(tagged_data,
                        total_examples=doc2vec_model.corpus_count,
                        epochs=args.doc2vec_epochs)
    
    doc2vec_model.save(str(model_path))

    return doc2vec_model

def visualize_author_diff(df: pd.DataFrame,
                          vectorized_stanzas: np.ndarray,
                          n_authors: int,
                          min_stanza_freq: int=15,
                          plot_dir: Path|None=None) -> None:

    pca = PCA(n_components=3)
    results = pca.fit_transform(vectorized_stanzas)
    
    #tsne = TSNE(n_components=3, perplexity=20, learning_rate=10, max_iter=2000)
    #results = tsne.fit_transform(vectorized_stanzas)

    df = df[~df["author"].str.contains("Anonymous", na=False)]

    author_counts = df["author"].value_counts()
    frequent_values = author_counts[author_counts > min_stanza_freq].index.tolist()
    sampled_authors = np.random.choice(frequent_values, size=n_authors)

    logger.info(f"Sampling {len(sampled_authors)} authors from total of {len(df["author"].unique())}")
    filtered_df = df[df["author"].isin(sampled_authors)]

    print(df.columns.tolist())

    def return_if_lower(x: pd.DataFrame): return x.sample(n=min(len(x), min_stanza_freq))
    df = filtered_df.groupby("author").apply(return_if_lower, include_groups=False)
    df = df.reset_index()
    print(df.columns.tolist())

    labels = df["author"].unique()

    colors = sns.color_palette("hls", len(labels))
    label_color_map = dict(zip(labels, colors))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    for label in labels:
        indices = [i for i, l in enumerate(df["author"]) if l == label]
        
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

    ax.set_title("PCA embeddings of documents (stanzas)", fontsize=15)
    ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/author_pca_3D.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 8))

    for label in labels:
        indices = [i for i, l in enumerate(df["author"]) if l == label]
        
        ax.scatter(
            results[indices,0],
            results[indices,1],
            label=label,
            color=label_color_map[label],
            s=60,
            alpha=.7,
            edgecolors="w"
        )

    ax.grid(alpha=.5)
    ax.set_title("PCA embeddings of documents (stanzas)", fontsize=15)
    ax.legend(title="Labels")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    plt.savefig(f"{plot_dir}/author_pca_2D.png")
    plt.show()
    plt.close()

def visualize_volume_diff(df: pd.DataFrame,
                          vectorized_stanzas: np.ndarray,
                          fraction: float,
                          plot_dir: Path|None=None) -> None:

    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(vectorized_stanzas)
    #tsne = TSNE(n_components=3, perplexity=20, learning_rate=10, max_iter=2000)
    #tsne_results = tsne.fit_transform(vectorized_stanzas)

    chosen_volumes = df[df["volume"].isin([1, 5, 7])]
    sampled_rows = chosen_volumes.sample(frac=fraction)
    logger.info(f"Sampling {sampled_rows.shape[0]} rows from total of {df.shape[0]}")
    df = sampled_rows

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    labels = df["volume"].unique()

    colors = sns.color_palette("hls", len(labels))
    label_color_map = dict(zip(labels, colors))

    for label in sorted(labels):
        indices = [i for i, l in enumerate(df["volume"]) if l == label]
        
        ax.scatter(
            pca_results[indices,0],
            pca_results[indices,1],
            pca_results[indices,2],
            label=label,
            color=label_color_map[label],
            s=60,
            alpha=.7,
            edgecolors="w"
        )

    ax.set_title("PCA embeddings of documents (stanzas)", fontsize=15)
    ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/volume_pca.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__=="__main__":
    args = parse_args()
    current_dir = Path(__file__).parent

    df = load_data(args)

    bin_dir = current_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    doc2vec_default = bin_dir / "doc2vec"
    if not args.overwrite_models and next(bin_dir.glob("doc2vec*"), None):
        doc2vec_model = Doc2Vec.load(str(doc2vec_default))
    else:
        doc2vec_model = train_doc2vec(df["tokenized"], doc2vec_default, args)

    tokenized = df["tokenized"]
    stanza_vectors = np.array([doc2vec_model.dv[i] for i in range(len(tokenized))])

    plot_dir = current_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    visualize_author_diff(df, stanza_vectors, args.n_authors, plot_dir=plot_dir)
    #visualize_volume_diff(df, stanza_vectors, args.volume_row_fraction, plot_dir=plot_dir)