import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd

dataset = pd.read_csv('data-kmeans.csv')
global data; data = dataset.values
global k; k=5

def compute_distance(a, b):
    return np.sqrt(np.sum(np.square(a-b), axis=-1))

def compute_centroid(Z):
    '''
    Z : a cluster set
    '''
    return np.average(Z, axis=0)

def compute_label(z, M):
    '''
    z : one data point
    M : a set of centroids of every clusters
    '''
    return np.argmin(compute_distance(z, M), axis=-1)

def compute_loss(C, M):
    '''
    C : a set of clusters (labels)
    M : a set of centroids of every clusters
    '''
    loss = 0
    for i in range(k):
        cluster = data[C==i]
        dists = compute_distance(cluster, M[i])
        loss += np.sum(dists)
    return loss/k

def random_initialize():
    labels = np.zeros(len(data))
    for i in range(len(data)):
        labels[i] = np.random.randint(k)
    return labels

import copy
def k_means_clustering():
    bst_initial = []
    bst_loss = []
    bst_centroids = []
    bst_labels = []
    for t in range(100):
        labels = random_initialize()
        initial = copy.deepcopy(labels)
        loss_it = []
        centroids_it = []
        break_cnt = 0
        while True:
            # calculate centroids
            centroids = []
            for i in range(k):
                centroids.append(compute_centroid(data[labels==i]))
            centroids = np.array(centroids)
            # update labels
            labels = np.zeros(len(data))
            for i in range(len(data)):
                labels[i] = compute_label(data[i,:], centroids)
            
            loss_it.append(compute_loss(labels, centroids))
            centroids_it.append
            if len(loss_it) > 1 and loss_it[-1]==loss_it[-2]:
                break_cnt += 1
            if break_cnt >10:
                break
        if t==0:    
            bst_initial = copy.deepcopy(initial)
            bst_loss = copy.deepcopy(loss_it)
            bst_centroids = copy.deepcopy(centroids_it)
            bst_labels = copy.deepcopy(labels)
        else:
            if bst_loss[-1] > loss_it[-1]:
                bst_initial = copy.deepcopy(initial)
                bst_loss = copy.deepcopy(loss_it)
                bst_centroids = copy.deepcopy(centroids_it)
                bst_labels = copy.deepcopy(labels)

    return bst_initial, bst_loss, bst_centroids, bst_labels


initial, loss, centroids, labels = k_means_clustering()

def plot_clusters(labels):
    color_it = iter(cm.rainbow(np.linspace(0,1,k)))
    for i in range(k):
        cluster = data[labels==i]
        color = next(color_it)
        plt.scatter(cluster[:,0], cluster[:,1], label='cluster :'+str(i), c=color)
    plt.show()

### RESULTS ###
# 01 Plot the data points
plt.scatter(data[:,0], data[:,1], label='data')
plt.title('data point')
plt.legend()
plt.show()

# 02 Visualize the initial condition of the point labels
plot_clusters(initial)

# 03 Plot the loss curve
plt.plot(loss)
plt.show()

# 04 Plot the centroid of each cluster

# 05 Plot the final clustering result
plot_clusters(labels)