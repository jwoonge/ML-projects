import numpy as np
import matplotlib.pyplot as plt
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

def k_means_clustering():
    labels = random_initialize()

    losses = []
    for it in range(100):
        # calculate centroids
        centroids = []
        for i in range(k):
            centroids.append(compute_centroid(data[labels==i]))
        centroids = np.array(centroids)
        # update labels
        labels = np.zeros(len(data))
        for i in range(len(data)):
            labels[i] = compute_label(data[i,:], centroids)
        # calculate loss
        loss = compute_loss(labels, centroids)
        losses.append(loss)
    return losses

loss = k_means_clustering()
plt.plot(loss)
plt.show()




'''
### RESULTS ###
# 01 Plot the data points
plt.scatter(data[:,0], data[:,1], label='data')
plt.title('data point')
plt.legend()
plt.show()

# 02 Visualize the initial condition of the point labels

# 03 Plot the loss curve

# 04 Plot the centroid of each cluster

# 05 Plot the final clustering result
'''