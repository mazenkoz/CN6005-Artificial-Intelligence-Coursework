"""CN6005 Task 2 - Text mining and K-Means clustering of job adverts (TF-IDF vs Sentence-BERT).
Run: python Task2_TextMining.py   (figures are saved to the screenshots/ folder)
"""

# <a href="https://colab.research.google.com/github/mazenkoz/artificial-intelligence-coursework/blob/main/Task2_TextMining.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Task 2: Text Mining and Document Clustering
# **CN6005 Artificial Intelligence – Group Coursework**
#
# Dataset: *Real / Fake Job Posting Prediction* (Bansal, Kaggle) – 17,880 job adverts, 866 of them labelled as fraudulent.
#
# Notebook sections:
# 1. Dataset loading and exploration (2.1)
# 2. Text pre-processing (2.2)
# 3. TF-IDF feature extraction (2.3)
# 4. K-Means clustering with the Elbow Method and Silhouette Score (2.4)
# 5. Sentence-BERT embeddings and comparison with TF-IDF (2.5)
# 6. Results summary (2.6)

# Install the required libraries
# %pip -q install kagglehub sentence-transformers nltk

import kagglehub

# Download the Real / Fake Job Posting dataset from Kaggle
dataset_path = kagglehub.dataset_download(
    "shivamb/real-or-fake-fake-jobposting-prediction"
)

print("Dataset downloaded to:")
print(dataset_path)

# Import the libraries
import re
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    adjusted_rand_score
)

# Download the NLTK resources used for pre-processing
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# Set random seeds
random.seed(42)
np.random.seed(42)

# Set a consistent graph style
sns.set_style("whitegrid")

# Create a folder for saving results
output_dir = Path("/content/task2_results")
output_dir.mkdir(parents=True, exist_ok=True)

print("Libraries loaded")

# ## 2.1 Dataset Selection and Exploration

# Find the CSV file inside the downloaded folder
csv_files = list(Path(dataset_path).rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file was found in the dataset folder.")

df = pd.read_csv(csv_files[0])

print("File loaded:", csv_files[0].name)
print("Dataset shape:", df.shape)
print("\nColumns:")
print(list(df.columns))

print(df.head())

# Check missing values in each column
missing = (df.isnull().mean() * 100).sort_values(ascending=False)

print("Percentage of missing values per column:")
print(missing.round(1))

plt.figure(figsize=(10, 5))

sns.barplot(
    x=missing.values,
    y=missing.index,
    color="steelblue"
)

plt.title("Missing Values per Column (%)")
plt.xlabel("Missing (%)")
plt.ylabel("Column")
plt.tight_layout()
plt.savefig(output_dir / "missing_values.png", dpi=150)
plt.close()

# Plot the number of real and fake job postings
fraud_counts = df["fraudulent"].value_counts().rename(
    {0: "Real", 1: "Fake"}
)

print("Real and fake postings:")
print(fraud_counts)
print("Fake postings (%):",
      round(fraud_counts["Fake"] / len(df) * 100, 2))

plt.figure(figsize=(6, 4))

sns.barplot(
    x=fraud_counts.index,
    y=fraud_counts.values,
    color="steelblue"
)

plt.title("Real vs Fake Job Postings")
plt.xlabel("Posting Type")
plt.ylabel("Number of Postings")
plt.tight_layout()
plt.savefig(output_dir / "real_vs_fake.png", dpi=150)
plt.close()

# Plot the ten most common job functions
top_functions = df["function"].value_counts().head(10)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=top_functions.values,
    y=top_functions.index,
    color="steelblue"
)

plt.title("Top 10 Job Functions")
plt.xlabel("Number of Postings")
plt.ylabel("Function")
plt.tight_layout()
plt.savefig(output_dir / "top_functions.png", dpi=150)
plt.close()

# Show the most common employment types
print("Employment types:")
print(df["employment_type"].value_counts(dropna=False))

# Each job posting is split across several text columns. They are combined into one document per posting so that the
# clustering uses the full advert (title, company profile, description, requirements and benefits).

# Combine the text columns into one document per job posting
text_columns = [
    "title", "company_profile", "description",
    "requirements", "benefits"
]

df["text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

# Remove exact duplicate documents
before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)

print("Documents before duplicate removal:", before)
print("Duplicate documents removed:", before - len(df))

# Count the words in each document
df["word_count"] = df["text"].str.split().str.len()

# Remove very short documents that contain little information
df = df[df["word_count"] >= 20].reset_index(drop=True)

print("Documents after removing very short adverts:", len(df))
print("\nWord count summary:")
print(df["word_count"].describe().round(1))

# Plot the document length distribution
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="word_count",
    hue=df["fraudulent"].map({0: "Real", 1: "Fake"}),
    bins=60,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("Document Length by Posting Type")
plt.xlabel("Words per Document")
plt.ylabel("Density")
plt.xlim(0, 1500)
plt.tight_layout()
plt.savefig(output_dir / "word_count_distribution.png", dpi=150)
plt.close()

print("Median words (real):",
      df.loc[df["fraudulent"] == 0, "word_count"].median())
print("Median words (fake):",
      df.loc[df["fraudulent"] == 1, "word_count"].median())

# A random sample of 5,000 postings is used for clustering. This keeps the Sentence-BERT encoding and the repeated
# K-Means runs manageable on Google Colab while still exceeding the 1,000-document requirement.

# Use a random sample for faster execution
sample_size = 5000

df = df.sample(
    n=min(sample_size, len(df)),
    random_state=42
).reset_index(drop=True)

assert len(df) >= 1000, "Dataset must contain at least 1,000 documents"

print("Documents used for clustering:", len(df))
print("Fake postings in the sample:", df["fraudulent"].sum())
print("\nDataset meets the coursework size requirement.")

# ## 2.2 Text Pre-processing

# Set up the stop-word list and lemmatiser
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def basic_clean(text):
    """Remove dataset placeholders and fix joined words."""

    # The dataset replaces links, emails and phones with tags like #URL_abc#
    text = re.sub(r"#(URL|EMAIL|PHONE)_[^#]*#", " ", text)

    # Remove any HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Split words that were joined when line breaks were removed
    # e.g. "our teamWe are" -> "our team We are"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Remove extra spaces
    return re.sub(r"\s+", " ", text).strip()


def preprocess(text):
    """Full pre-processing pipeline used for TF-IDF."""

    # Clean placeholders and joined words
    text = basic_clean(text)

    # Lowercase the text
    text = text.lower()

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Split the text into tokens
    tokens = word_tokenize(text)

    # Remove stop words and very short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Reduce each word to its dictionary form
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# Apply the pre-processing
start_time = time.time()

df["clean_text"] = df["text"].apply(preprocess)

print("Pre-processing time (seconds):",
      round(time.time() - start_time, 2))

# Show each pre-processing step for one example document
example = df["text"].iloc[0][:400]

step1 = basic_clean(example)
step2 = step1.lower()
step3 = re.sub(r"[^a-z\s]", " ", step2)
step4 = word_tokenize(step3)
step5 = [t for t in step4 if t not in stop_words and len(t) > 2]
step6 = [lemmatizer.lemmatize(t) for t in step5]

steps = pd.DataFrame({
    "Step": [
        "Original text",
        "Placeholders removed / joined words split",
        "Lowercasing",
        "Punctuation and special characters removed",
        "Tokenisation",
        "Stop-word removal",
        "Lemmatisation"
    ],
    "Output": [
        example[:200],
        step1[:200],
        step2[:200],
        step3[:200],
        str(step4[:25]),
        str(step5[:25]),
        str(step6[:25])
    ]
})

pd.set_option("display.max_colwidth", 200)
print(steps)

steps.to_csv(output_dir / "preprocessing_example.csv", index=False)

# Show examples of lemmatisation
for word in ["services", "companies", "managing", "skills", "studies"]:
    print(f"{word:>10} -> {lemmatizer.lemmatize(word)}")

# Compare document length before and after pre-processing
df["clean_word_count"] = df["clean_text"].str.split().str.len()

print("Average words before:", round(df["word_count"].mean(), 1))
print("Average words after:", round(df["clean_word_count"].mean(), 1))
print("Reduction (%):",
      round((1 - df["clean_word_count"].mean() / df["word_count"].mean()) * 100, 1))

# ## 2.3 Feature Extraction Using TF-IDF

# Generate TF-IDF vectors
tfidf = TfidfVectorizer(
    max_features=5000,     # keep the 5,000 most frequent terms
    min_df=5,              # ignore terms in fewer than 5 documents
    max_df=0.8,            # ignore terms in more than 80% of documents
    ngram_range=(1, 2)     # use single words and two-word phrases
)

X_tfidf = tfidf.fit_transform(df["clean_text"])
terms = np.array(tfidf.get_feature_names_out())

print("TF-IDF matrix shape:", X_tfidf.shape)
print("Matrix sparsity (%):",
      round((1 - X_tfidf.nnz / np.prod(X_tfidf.shape)) * 100, 2))

# Identify the key terms with the highest average TF-IDF weight
mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
top_index = mean_tfidf.argsort()[::-1][:20]

top_terms = pd.DataFrame({
    "Term": terms[top_index],
    "Mean TF-IDF": mean_tfidf[top_index]
})

print(top_terms.round(4))

plt.figure(figsize=(9, 6))

sns.barplot(
    data=top_terms,
    x="Mean TF-IDF",
    y="Term",
    color="steelblue"
)

plt.title("Top 20 Terms by Average TF-IDF Weight")
plt.tight_layout()
plt.savefig(output_dir / "top_tfidf_terms.png", dpi=150)
plt.close()

top_terms.to_csv(output_dir / "top_tfidf_terms.csv", index=False)

# Compare key terms in real and fake postings
real_mask = (df["fraudulent"] == 0).values
fake_mask = (df["fraudulent"] == 1).values

real_mean = np.asarray(X_tfidf[real_mask].mean(axis=0)).ravel()
fake_mean = np.asarray(X_tfidf[fake_mask].mean(axis=0)).ravel()

difference = fake_mean - real_mean

term_compare = pd.DataFrame({
    "More common in fake postings": terms[difference.argsort()[::-1][:15]],
    "More common in real postings": terms[difference.argsort()[:15]]
})

print(term_compare)
term_compare.to_csv(output_dir / "real_vs_fake_terms.csv", index=False)

# ## 2.4 Document Clustering Using K-Means
#
# TF-IDF vectors have thousands of sparse dimensions, which makes distances between documents less meaningful.
# Truncated SVD (Latent Semantic Analysis) is used to reduce them to 50 dimensions, matching the 50 PCA components
# used later for the Transformer embeddings so both methods are clustered in a space of the same size.

# Reduce the TF-IDF vectors to 50 dimensions
svd = TruncatedSVD(n_components=50, random_state=42)

X_tfidf_svd = normalize(svd.fit_transform(X_tfidf))

print("Reduced TF-IDF shape:", X_tfidf_svd.shape)
print("Variance explained by 50 components (%):",
      round(svd.explained_variance_ratio_.sum() * 100, 2))

def evaluate_k_values(X, k_values):
    """Run K-Means for several K values and record inertia and silhouette."""

    rows = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        rows.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette_score(X, labels)
        })

    return pd.DataFrame(rows)


def plot_k_selection(results, title, filename):
    """Plot the Elbow Method and Silhouette Score side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(results["K"], results["Inertia"], marker="o")
    axes[0].set_title(f"Elbow Method ({title})")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(results["K"], results["Silhouette"],
                 marker="o", color="darkorange")
    axes[1].set_title(f"Silhouette Score ({title})")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


# Test K values from 2 to 12
k_values = range(2, 13)

start_time = time.time()
tfidf_k_results = evaluate_k_values(X_tfidf_svd, k_values)

print("K selection time (seconds):", round(time.time() - start_time, 2))
print(tfidf_k_results.round(4))

plot_k_selection(tfidf_k_results, "TF-IDF", "tfidf_k_selection.png")

# Choose K using the highest silhouette score.
# K = 2 is excluded because two very broad groups are not useful
# for describing the different types of job advert.
candidates = tfidf_k_results[tfidf_k_results["K"] >= 3]
best_k_tfidf = int(candidates.loc[candidates["Silhouette"].idxmax(), "K"])

print("Selected K for TF-IDF:", best_k_tfidf)

# Apply K-Means with the selected K
kmeans_tfidf = KMeans(
    n_clusters=best_k_tfidf,
    random_state=42,
    n_init=20
)

df["tfidf_cluster"] = kmeans_tfidf.fit_predict(X_tfidf_svd)

cluster_counts = df["tfidf_cluster"].value_counts().sort_index()

print("Documents per TF-IDF cluster:")
print(cluster_counts)

def top_terms_per_cluster(cluster_labels, n_terms=10):
    """Return the highest average TF-IDF terms for each cluster."""

    rows = []

    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        centre = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
        best = centre.argsort()[::-1][:n_terms]

        rows.append({
            "Cluster": cluster,
            "Documents": int(mask.sum()),
            "Fake (%)": round(df.loc[mask, "fraudulent"].mean() * 100, 2),
            "Top Terms": ", ".join(terms[best])
        })

    return pd.DataFrame(rows)


# Interpret each TF-IDF cluster using its key terms
tfidf_cluster_terms = top_terms_per_cluster(df["tfidf_cluster"].values)

print(tfidf_cluster_terms)
tfidf_cluster_terms.to_csv(output_dir / "tfidf_cluster_terms.csv", index=False)

# Show example job titles from each cluster
for cluster in sorted(df["tfidf_cluster"].unique()):

    print("\n" + "=" * 70)
    print(f"TF-IDF CLUSTER {cluster}")
    print("=" * 70)

    sample_titles = df[df["tfidf_cluster"] == cluster]["title"].head(8)

    for title in sample_titles:
        print("-", title)

def plot_clusters_2d(X, cluster_labels, title, filename):
    """Project the vectors to two dimensions with PCA and plot the clusters."""

    points = PCA(n_components=2, random_state=42).fit_transform(X)

    plot_df = pd.DataFrame({
        "PC1": points[:, 0],
        "PC2": points[:, 1],
        "Cluster": cluster_labels.astype(str)
    })

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        alpha=0.6,
        s=15
    )

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


plot_clusters_2d(
    X_tfidf_svd,
    df["tfidf_cluster"].values,
    "TF-IDF K-Means Clusters (PCA projection)",
    "tfidf_clusters_pca.png"
)

def plot_cluster_vs_function(cluster_column, title, filename):
    """Show how the clusters relate to the job function column."""

    labelled = df.dropna(subset=["function"])
    top_function_names = labelled["function"].value_counts().head(8).index
    labelled = labelled[labelled["function"].isin(top_function_names)]

    ct = pd.crosstab(labelled[cluster_column], labelled["function"])

    plt.figure(figsize=(12, 6))

    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")

    plt.title(title)
    plt.xlabel("Job Function (dataset label)")
    plt.ylabel("Cluster")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


plot_cluster_vs_function(
    "tfidf_cluster",
    "TF-IDF Cluster Membership by Job Function",
    "tfidf_cluster_vs_function.png"
)

def plot_fake_rate(cluster_column, title, filename):
    """Plot the percentage of fake postings in each cluster."""

    fake_rate = df.groupby(cluster_column)["fraudulent"].mean() * 100
    overall = df["fraudulent"].mean() * 100

    plt.figure(figsize=(8, 4.5))

    sns.barplot(
        x=fake_rate.index.astype(str),
        y=fake_rate.values,
        color="indianred"
    )

    plt.axhline(overall, color="black", linestyle="--",
                label=f"Overall rate ({overall:.1f}%)")

    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Fake Postings (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


plot_fake_rate(
    "tfidf_cluster",
    "Fake Postings per TF-IDF Cluster",
    "tfidf_fake_rate.png"
)

# ## 2.5 Transformer-Based Embeddings and Comparison
#
# Sentence-BERT (`all-MiniLM-L6-v2`) produces a 384-dimensional embedding for each document. The Transformer receives
# the lightly cleaned text rather than the lemmatised text, because it was trained on natural sentences and uses word
# order and context. The model reads up to 256 word-pieces, so longer adverts are truncated.

from sentence_transformers import SentenceTransformer

# Load the pre-trained Sentence-BERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Maximum sequence length:", sbert_model.max_seq_length)

# Lightly clean the text for the Transformer
transformer_text = df["text"].apply(basic_clean).tolist()

# Generate the embeddings
start_time = time.time()

embeddings = sbert_model.encode(
    transformer_text,
    show_progress_bar=True,
    batch_size=64
)

sbert_time = time.time() - start_time

print("Embedding shape:", embeddings.shape)
print("Embedding time (seconds):", round(sbert_time, 2))

# Normalise the embeddings and reduce them to 50 dimensions
embeddings = normalize(embeddings)

pca = PCA(n_components=50, random_state=42)
X_sbert_pca = normalize(pca.fit_transform(embeddings))

print("Reduced embedding shape:", X_sbert_pca.shape)
print("Variance explained by 50 components (%):",
      round(pca.explained_variance_ratio_.sum() * 100, 2))

# Find a suitable K for the Transformer embeddings
start_time = time.time()
sbert_k_results = evaluate_k_values(X_sbert_pca, k_values)

print("K selection time (seconds):", round(time.time() - start_time, 2))
print(sbert_k_results.round(4))

plot_k_selection(sbert_k_results, "Sentence-BERT", "sbert_k_selection.png")

candidates = sbert_k_results[sbert_k_results["K"] >= 3]
best_k_sbert = int(candidates.loc[candidates["Silhouette"].idxmax(), "K"])

print("Selected K for Sentence-BERT:", best_k_sbert)

# Cluster the embeddings with K-Means
kmeans_sbert = KMeans(
    n_clusters=best_k_sbert,
    random_state=42,
    n_init=20
)

df["sbert_cluster"] = kmeans_sbert.fit_predict(X_sbert_pca)

print("Documents per Sentence-BERT cluster:")
print(df["sbert_cluster"].value_counts().sort_index())

# Interpret the clusters using TF-IDF key terms
sbert_cluster_terms = top_terms_per_cluster(df["sbert_cluster"].values)

print(sbert_cluster_terms)
sbert_cluster_terms.to_csv(output_dir / "sbert_cluster_terms.csv", index=False)

# Show example job titles from each Sentence-BERT cluster
for cluster in sorted(df["sbert_cluster"].unique()):

    print("\n" + "=" * 70)
    print(f"SENTENCE-BERT CLUSTER {cluster}")
    print("=" * 70)

    sample_titles = df[df["sbert_cluster"] == cluster]["title"].head(8)

    for title in sample_titles:
        print("-", title)

plot_clusters_2d(
    X_sbert_pca,
    df["sbert_cluster"].values,
    "Sentence-BERT K-Means Clusters (PCA projection)",
    "sbert_clusters_pca.png"
)

plot_cluster_vs_function(
    "sbert_cluster",
    "Sentence-BERT Cluster Membership by Job Function",
    "sbert_cluster_vs_function.png"
)

plot_fake_rate(
    "sbert_cluster",
    "Fake Postings per Sentence-BERT Cluster",
    "sbert_fake_rate.png"
)

# ### Comparing cluster quality
#
# Three **internal** measures (no labels needed) and two **external** measures are used:
#
# * **Silhouette Score** (−1 to 1, higher is better): how close documents are to their own cluster compared with the nearest other cluster.
# * **Calinski-Harabasz Index** (higher is better): ratio of between-cluster to within-cluster spread.
# * **Davies-Bouldin Index** (lower is better): average similarity between each cluster and its most similar cluster.
# * **NMI** and **ARI** against the dataset's `function` column (e.g. Sales, Engineering): how well the clusters match real job areas. This column was not used for clustering.
#
# To make the comparison fair, both methods are also evaluated with the same number of clusters.

def cluster_quality(X, cluster_labels, method, k):
    """Calculate internal and external cluster quality measures."""

    labelled = df["function"].notna().values

    return {
        "Method": method,
        "K": k,
        "Silhouette": silhouette_score(X, cluster_labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, cluster_labels),
        "Davies-Bouldin": davies_bouldin_score(X, cluster_labels),
        "NMI (function)": normalized_mutual_info_score(
            df.loc[labelled, "function"], cluster_labels[labelled]
        ),
        "ARI (function)": adjusted_rand_score(
            df.loc[labelled, "function"], cluster_labels[labelled]
        )
    }


# Use the same K for both methods for a direct comparison
common_k = best_k_tfidf

sbert_same_k = KMeans(
    n_clusters=common_k, random_state=42, n_init=20
).fit_predict(X_sbert_pca)

comparison = pd.DataFrame([
    cluster_quality(X_tfidf_svd, df["tfidf_cluster"].values,
                    "TF-IDF + K-Means", best_k_tfidf),
    cluster_quality(X_sbert_pca, sbert_same_k,
                    "Sentence-BERT + K-Means (same K)", common_k),
    cluster_quality(X_sbert_pca, df["sbert_cluster"].values,
                    "Sentence-BERT + K-Means (own best K)", best_k_sbert)
])

print("TF-IDF and Sentence-BERT Clustering Comparison")
print(comparison.round(4))

comparison.to_csv(output_dir / "clustering_comparison.csv", index=False)

# Measure how similar the two sets of clusters are
agreement = adjusted_rand_score(df["tfidf_cluster"], sbert_same_k)
print("\nAgreement between TF-IDF and Sentence-BERT clusters (ARI):",
      round(agreement, 4))

# Plot the comparison for the same number of clusters
plot_metrics = comparison.iloc[:2].melt(
    id_vars="Method",
    value_vars=["Silhouette", "Davies-Bouldin", "NMI (function)", "ARI (function)"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(11, 5))

sns.barplot(
    data=plot_metrics,
    x="Metric",
    y="Score",
    hue="Method"
)

plt.title(f"TF-IDF vs Sentence-BERT Clustering (K = {common_k})")
plt.ylabel("Score")
plt.xlabel("")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(output_dir / "tfidf_vs_sbert.png", dpi=150)
plt.close()

print("Note: for Davies-Bouldin a lower score is better; for the others higher is better.")

# ## 2.6 Results Summary

# Summarise the main findings for the report
tfidf_row = comparison.iloc[0]
sbert_row = comparison.iloc[1]

print("Documents clustered:", len(df))
print("TF-IDF vocabulary size:", X_tfidf.shape[1])
print("Sentence-BERT embedding size:", embeddings.shape[1])
print("Sentence-BERT encoding time (s):", round(sbert_time, 2))

print("\nSelected K  -> TF-IDF:", best_k_tfidf, "| Sentence-BERT:", best_k_sbert)

for metric, higher_better in [
    ("Silhouette", True),
    ("Calinski-Harabasz", True),
    ("Davies-Bouldin", False),
    ("NMI (function)", True),
    ("ARI (function)", True)
]:
    t, s = tfidf_row[metric], sbert_row[metric]
    if np.isclose(t, s):
        better = "equal"
    else:
        better = "Sentence-BERT" if (s > t) == higher_better else "TF-IDF"
    print(f"{metric:<18} TF-IDF={t:.4f}  SBERT={s:.4f}  -> better: {better}")

# Highest fake rate found by each method
for column in ["tfidf_cluster", "sbert_cluster"]:
    rate = df.groupby(column)["fraudulent"].mean() * 100
    print(f"\n{column}: highest fake rate {rate.max():.1f}% in cluster {rate.idxmax()}")

# Save the clustered dataset
df[["title", "function", "fraudulent", "tfidf_cluster", "sbert_cluster"]].to_csv(
    output_dir / "clustered_job_postings.csv", index=False
)
print("\nResults saved to", output_dir)
