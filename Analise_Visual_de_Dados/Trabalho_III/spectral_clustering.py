import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(X, n_clusters, n_neighbors, sigma):
    # Step 1: Compute the affinity matrix
    affinity_matrix = compute_affinity_matrix(X, n_neighbors, sigma)

    # Step 2: Construct the Laplacian matrix
    laplacian_matrix = construct_laplacian_matrix(affinity_matrix)

    # Step 3: Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4: Select the eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :n_clusters]

    # Step 5: Cluster the data
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(selected_eigenvectors)
    labels = kmeans.labels_

    return labels

def compute_affinity_matrix(X, n_neighbors, sigma):
    # Compute pairwise distances
    pairwise_distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)

    # Find nearest neighbors
    nearest_neighbors_indices = np.argsort(pairwise_distances, axis=-1)[:, 1:(n_neighbors+1)]

    # Compute affinity matrix using the Gaussian kernel
    affinity_matrix = np.exp(-pairwise_distances**2 / (2 * sigma**2))
    affinity_matrix[np.arange(len(X))[:, np.newaxis], nearest_neighbors_indices] = 1  # Fully connected within neighbors
    affinity_matrix[nearest_neighbors_indices, np.arange(len(X))] = 1  # Fully connected within neighbors

    return affinity_matrix

def construct_laplacian_matrix(affinity_matrix):
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    laplacian_matrix = degree_matrix - affinity_matrix
    return laplacian_matrix