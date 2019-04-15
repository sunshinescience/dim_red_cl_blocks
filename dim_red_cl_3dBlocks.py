'''
-----------------------------------------------------------------
A demo of dimensionality reduction and clustering on 3D data
-----------------------------------------------------------------
'''

import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation

# Load the dataset
data = np.load("tensors_OLYMPUS1.npy")
# data = np.load("enc_v7.npy") # reduced data
tensors = data

# standardize the data 
data = data.reshape((data.shape[0], -1))
data = scale(data)

# Number of samples
n_samples = data.data.shape[0]

# #############################################################################
# Plot of one block
if 1:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    # tensors is what you load from tensors = np.load("tensors.npy") or similar
    # and 0 is the index of the block to visualize
    tensor = tensors[1]
    voxels = tensor > 0.0
    # ax.set_title(labels[i])
    ax.voxels(voxels, edgecolor='k', alpha=0.5)

# Plot of eight blocks
# indexes is the number of blocks to plot, which should be a number greater than or equal to 4
if 1:   
    indexes = 8
    fig_size = (10,8)

    n_rand=[]
    for i in range(indexes):
        n_rand.append(np.random.randint(0, n_samples))
    print (n_rand)

    fig = plt.figure(figsize=fig_size)
    plt_index = 0
    for i in n_rand:
        tensor = tensors[i]
        voxels = tensor > 0.0
        plt_index = plt_index + 1
        ax = fig.add_subplot((indexes-(indexes/2)), 2, plt_index, projection='3d')
        ax.voxels(voxels, facecolors='g', edgecolor='k', alpha=0.5)
    plt.tight_layout()
    plt.show()

# #############################################################################
# t-SNE reduced data

n_iter = 1000
n_perplexity = 40

t_sne = TSNE(n_components=2, perplexity=n_perplexity, n_iter=n_iter)
tsne_result = t_sne.fit_transform(data) # t-SNE reduced data

# #############################################################################
# Truncated SVD reduced data

svd = TruncatedSVD(n_components=2, n_iter=10)
svd_result = svd.fit_transform(data)  

# #############################################################################
# Isomap reduced data

iso = Isomap(n_components=2)
iso_result = iso.fit_transform(data)

# #############################################################################
# Pca reduced data

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# #############################################################################
# K-means clustering
# Limitations:
#   Note that K-means is limited to linear cluster boundaries - if the clusters have 
#   complicated geometries, K-means may not be effective

n_clusters = 18
n_init = 10

def k_means_reduced(reduced_data, initialization, n_clusters, n_init):
    """
    This returns K-means clustering on data that has undergone dimensionality reduction.
    Parameters:
        reduced_data: The data that has undergone dimensionality reduction
        initialization: Method for initialization, defaults to ‘k-means++’:
        n_clusters: The number of clusters to form as well as the number of centroids to generate.
        n_init: Number of times the k-means algorithm will run with different centroid seeds.
    """
    k_means = KMeans(init=initialization, n_clusters=n_clusters, n_init=n_init) 
    k_means_model = k_means.fit(reduced_data)
    return k_means_model

# K-means clustering on t-SNE reduced data
k_t_sne = k_means_reduced(tsne_result, 'k-means++', n_clusters, n_init)
# K-means clustering on Truncated SVD reduced data
k_SVD = k_means_reduced(svd_result, 'k-means++', n_clusters, n_init)
# K-means clustering on isomap reduced data
k_iso = k_means_reduced(iso_result, 'k-means++', n_clusters, n_init)
# K-means clustering on PCA reduced data
k_pca = k_means_reduced(pca_result, 'k-means++', n_clusters, n_init)

# #########
# Visualize random blocks from a chosen cluster

clust_num = 1 # The cluster number to select
indexes = 8 # The number of plots to generate on the figure. This should be an even number equal to or greater than 4.
fig_size = (10,8)

def cluster_indices(clust_num, labels_array): #numpy 
    """
    This takes parameters such as the cluster number (clust_num) and the labels  of each data point.
    This returns the indices of the cluster_num you provide.
    """
    return np.where(labels_array == clust_num)[0]

cluster_data = cluster_indices(clust_num, k_t_sne.labels_)
# print ('Samples from cluster {}:'.format(clust_num), cluster_data) # Prints sample numbers for the select cluster

# Plot of random blocks from a chosen cluster
cluster_data_rand = []
for i in range(indexes):
    rand_num = np.random.choice(cluster_data)
    cluster_data_rand.append(rand_num)
if 1:
    fig = plt.figure(figsize=fig_size)
    plt_index = 0
    for i in cluster_data_rand:
        tensor = tensors[i]
        voxels = tensor > 0.0
        plt_index = plt_index + 1
        ax = fig.add_subplot((indexes-(indexes/2)), 2, plt_index, projection='3d')
        ax.voxels(voxels, facecolors='g', edgecolor='k', alpha=0.5)
    fig_title = 'Random blocks from cluster ' + str(clust_num)
    fig.suptitle(fig_title, fontsize=12)
    plt.tight_layout()
    plt.show()

# Plot of the first block from the above figure  
if 0:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    # tensors is what you load from tensors = np.load("tensors.npy") or similar
    # and 0 is the index of the block to visualize
    tensor = tensors[cluster_data_rand[0]] # index 0 of the blocks from the above plot
    voxels = tensor > 0.0
    # ax.set_title(labels[i])
    ax.voxels(voxels, edgecolor='k', alpha=0.5)
    plt.show

def elbow_curve(dim_red_data):
    """
    Plots an elbow curve to select the optimal number of clusters (k) for k-means clustering.
    Fit KMeans and calculate sum of squared errors (SSE) for each cluster (k), which is 
      defined as the sum of the squared distance between each member of the cluster and its centroid.
    Parameter:
        dim_red_data: Dimensionality reduced data
    """
    sse = {}
    for k in range(1, 40): 
        # Initialize KMeans with k clusters and fit it 
        kmeans = KMeans(n_clusters=k, random_state=0).fit(dim_red_data)  
        # Assign sum of squared errors to k element of the sse dictionary
        sse[k] = kmeans.inertia_ 
    # Add the plot title, x and y axis labels
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    # Plot SSE values for each k stored as keys in the dictionary
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()

elbow_curve(tsne_result)

# ###############################################################
# Visualize the results of clustering on dimesnionality reduced data

def plot_dim_red_clust(dim_red, cluster_type, dim_red_name=None, cluster_type_name=None, centroids=None):
    """
    Visualize the results of clustering on dimesnionality reduced data.
    Plots the dimensionality reduced data vs. the clustered dimensionality reduced data
    Parameters:
        dim_red: dimensionality reduction method
        cluster_type: the type of clustering algorithm, for example, from the Scikit-learn library, one can use the k-means implementatation as sklearn.cluster.KMeans()
        dim_red_name: name of dimesnionality reduction to be in the title of the corresponding plot
        cluster_type_name: name of clustering method to be in the title of the corresponding plot
        centroids: plots the coordinates of cluster centers 
    """
    fig, axarr = plt.subplots(2, 1, figsize=(6,6))  
    # Plot of dimensionality reduced data 
    ax1 = axarr[0]
    ax1.scatter(dim_red[:,0], dim_red[:,1], c='k', marker='.')
    if dim_red_name:
        ax1.set_title('{} reduced data'.format(dim_red_name))
    # Plot of clustering on dimensionality reduced data
    ax2 = axarr[1]
    ax2.scatter(dim_red[:,0], dim_red[:,1], c=cluster_type.labels_, marker='.')
    if centroids:
        centroids = cluster_type.cluster_centers_
        ax2.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=100, linewidths=5,
                    c='k', zorder=10)
    if cluster_type_name:
        ax2.set_title('{} clustering on {} reduced data'.format(cluster_type_name, dim_red_name))
    plt.tight_layout()
    plt.show()

# Plot of k-means clustering on t-SNE reduced data
plot_dim_red_clust(tsne_result, k_t_sne, 't-SNE', 'K-means', centroids=True)
# Plot of k-means clustering on pca reduced data
plot_dim_red_clust(pca_result, k_pca, 'PCA', 'K-means')
# Plot of K-means clustering on isomap reduced data
plot_dim_red_clust(iso_result, k_iso, 'isomap', 'K-means')
# Plot of K-means clustering on truncated SVD reduced data
plot_dim_red_clust(svd_result, k_SVD, 'truncated SVD', 'K-means')

# #############################################################################
# Mean-shift clustering
# We can use the estimate_bandwidth function to estimate a good bandwidth for our data
bandwidth = round(estimate_bandwidth(data))

ms = MeanShift(bandwidth=bandwidth)
ms_tsne = ms.fit(tsne_result)
ms_labels = ms_tsne.labels_
ms_labels_unique = np.unique(ms_labels)
ms_n_clusters = len(ms_labels_unique)
print ('The number of estimated clusters from mean-shift clustering is: {}'.format(ms_n_clusters))

# Visualize the results of Mean-shift clustering with t-sne reduced data
plot_dim_red_clust(tsne_result, ms_tsne, 't-SNE', 'Mean-shift') # ****** Get this working ********

# #############################################################################
# Spectral clustering

sc_tsne = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                           assign_labels='discretize').fit(tsne_result)  
sc_iso = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                           assign_labels='discretize').fit(iso_result)  

# #########
# Plot the results of spectral clustering on t-SNE reduced data
plot_dim_red_clust(tsne_result, sc_tsne, 't-SNE', 'Spectral') 

# Plot the results of spectral clustering on isomap reduced data
plot_dim_red_clust(iso_result, sc_iso, 'isomap', 'Spectral')

# #############################################################################
# DBSCAN clustering

db = DBSCAN(eps=3, min_samples=2)
db_tsne = db.fit(tsne_result)

# #########
# Visualize the results of DBSCAN clustering
plot_dim_red_clust(tsne_result, db_tsne, 't-SNE', 'DBSCAN') # ****** Get this working ********

# #############################################################################
# Affinity propogation clustering

ap = AffinityPropagation().fit(tsne_result)

# #########
# Visualize the results of Affinity propogation clustering on t-SNE reduced data
plot_dim_red_clust(tsne_result, ap, 't-SNE', 'Affinity propogation', centroids=True) 
