
'''
Written by Chloe D. Kwon (dk837@cornell.edu/chloe.dkkwon@gmail.com)
March 7, 2025
How to run: python cluster_phone.py ../_data/wav ../_data
Objective: MFCC in each audio file are segmented into corresponding phones

Input: dictionary (pickle file) containing each phone label as a key and values as a list with each mfcc array
   e.g., ph_data['phone_A'] = [array 1 (shape=(2, 13)), array2 (shape=(5, 13)), ...]
Output: Dendrogram from hierarchical clustering and a scatter plot from KMeans clustering using t-SNE
   Prints evaluation results (cross-validation with 3 folds; adjust with `n_splits`)
'''

import pickle
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cluster_phone.py <phone_data_directory>")
        sys.exit(1)

    data_dir = sys.argv[1]

''' Load data '''
with open(data_dir, 'rb') as hdl:
    ph_data = pickle.load(hdl)

''' Pad each token to get a fixed length '''
# Find max len of the arrays for padding
max_frames = max(instance.shape[0] for instances in ph_data.values() for instance in instances)

# Pad all tokens to the max length
token_features = []
token_labels = []

for phoneme, instances in ph_data.items():
    for instance in instances:
        # Pad with zeros if the token is shorter than max_frames
        padded_instance = np.pad(instance, ((0, max_frames - instance.shape[0]), (0, 0)), mode='constant')

        # Flatten the (max_frames, 13) matrix into a 1D vector -> resulting in (1, max_frames*13)
        token_features.append(padded_instance.flatten())
        token_labels.append(phoneme)

token_features = np.array(token_features) 

# Get the number of unique phone labels
keys = list(ph_data.keys())
n_ph = len(set(keys))


''' Encode labels into index numbers '''
label_encoder = LabelEncoder()
true_label_numbers = label_encoder.fit_transform(token_labels)  # Convert phoneme labels to integers


''' Run Hierarchical clustering '''
h = AgglomerativeClustering(n_clusters=n_ph, metric = "euclidean", linkage="ward")
h_labs = h.fit_predict(token_features)


''' Visualize dendrogram for hierarchical clustering'''
plt.figure(figsize=(12, 18))
linkage_matrix = linkage(token_features, method="ward")
dendrogram(linkage_matrix,
           orientation="right", labels=token_labels,
           leaf_font_size=8, truncate_mode="level", p=10)  # Show top 10 levels
plt.title("Dendrogram of Sound Clustering")
plt.xlabel("Distance")
plt.ylabel("Label")
plt.savefig("dendrogram.pdf", dpi=300, bbox_inches="tight")
plt.show()


''' Run t-SNE for visualizing clustering results (to plot in 2D) '''
tsne = TSNE(n_components=2, perplexity=20, random_state=1)
tsne_features = tsne.fit_transform(token_features)


''' KMeans clustering '''
kmeans = KMeans(n_clusters=n_ph, random_state=1)
cluster_labels = kmeans.fit_predict(token_features)

# Create mapping of cluster -> most common phone
cluster_to_phoneme = {}
for cluster in np.unique(cluster_labels):
    phoneme_indices = true_label_numbers[cluster_labels == cluster]  # Get phone labels in this cluster
    most_common_phoneme = label_encoder.inverse_transform([Counter(phoneme_indices).most_common(1)[0][0]])[0]
    cluster_to_phoneme[cluster] = most_common_phoneme  # Store most common phone per cluster

# Plot cluster results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=cluster_labels, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, label="Cluster Label")
plt.title("Visualization of Phoneme Clustering (Padded Features)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")

# Draw Convex Hulls around clusters
for cluster in np.unique(cluster_labels):
    # Get all points belonging to this cluster
    points = tsne_features[cluster_labels == cluster]

    if len(points) > 2:  # ConvexHull needs at least 3 points
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], "k-", alpha=0.6)  # Black boundary lines

        # centroid for labeling
        centroid = np.mean(points, axis=0)
        phoneme_label = cluster_to_phoneme[cluster]
        plt.text(centroid[0], centroid[1], phoneme_label, fontsize=12, weight="bold",
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="black", boxstyle="round"))
plt.savefig("kmeans.png", dpi=300, bbox_inches="tight")
plt.show(block=True)


''' Cross-validate '''
n_splits = 3 # adjust based on data size
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
ari_scores, nmi_scores, f1_scores = [], [], []

for train_index, test_index in skf.split(token_features, true_label_numbers):
    # Split data
    X_train, X_test = token_features[train_index], token_features[test_index]
    y_train, y_test = true_label_numbers[train_index], true_label_numbers[test_index]

    kmeans = KMeans(n_clusters=n_ph, random_state=1, n_init=10)
    train_cluster_labels = kmeans.fit_predict(X_train)
    test_cluster_labels = kmeans.predict(X_test)

    ari = adjusted_rand_score(y_test, test_cluster_labels)
    nmi = normalized_mutual_info_score(y_test, test_cluster_labels)
    confusion = np.zeros((n_ph, n_ph))
    for i in range(len(y_test)):
        confusion[y_test[i], test_cluster_labels[i]] += 1

    row_ind, col_ind = linear_sum_assignment(confusion.max() - confusion)  # Hungarian Matching
    best_mapping = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_predictions = np.array([best_mapping[cluster] for cluster in test_cluster_labels])

    f1 = f1_score(y_test, mapped_predictions, average="weighted")  # Weighted to account for imbalance

    # Store results
    ari_scores.append(ari)
    nmi_scores.append(nmi)
    f1_scores.append(f1)

    print(f"Fold Results: ARI={ari:.4f}, NMI={nmi:.4f}, F1={f1:.4f}")

# Compute mean scores across folds
print("\nCross-Validation Results:")
print(f"Mean ARI: {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")
print(f"Mean NMI: {np.mean(nmi_scores):.4f} ± {np.std(nmi_scores):.4f}")
print(f"Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
