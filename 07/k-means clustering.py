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
    return loss/len(data)

def random_initialize():
    labels = np.zeros(len(data))
    for i in range(len(data)):
        labels[i] = np.random.randint(k)
    return labels

def check_valid_initial(label):
    cnt = [0 for x in range(k)]
    for l in label:
        cnt[int(l)] += 1
    print(cnt)
    for i in range(k):
        if cnt[i]==0:
            return False
    return True

import copy
def k_means_clustering():
    bst_loss = []
    bst_centroids = []
    bst_labels = []
    
    for t in range(100):
        labels = random_initialize()
        if check_valid_initial(labels):
            labels_it = [copy.deepcopy(labels)]
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
                centroids_it.append(centroids)
                labels_it.append(labels)
                if len(loss_it) > 1 and loss_it[-1]==loss_it[-2]:
                    break_cnt += 1
                if break_cnt >10:
                    break
            if t==0:
                bst_loss = copy.deepcopy(loss_it)
                bst_centroids = copy.deepcopy(centroids_it)
                bst_labels = copy.deepcopy(labels_it)
            else:
                if bst_loss[-1] > loss_it[-1]:
                    bst_loss = copy.deepcopy(loss_it)
                    bst_centroids = copy.deepcopy(centroids_it)
                    bst_labels = copy.deepcopy(labels_it)
    bst_centroids = np.array(bst_centroids)
    bst_centroids_distance = compute_distance([0,0], bst_centroids)
    return bst_labels, bst_loss, bst_centroids, bst_centroids_distance


labels, loss, centroids, centroid_distances = k_means_clustering()

def plot_clusters(labels, centroid):
    color_it = iter(cm.rainbow(np.linspace(0,1,k)))
    for i in range(k):
        cluster = data[labels==i]
        color = next(color_it).reshape(1,-1)
        plt.scatter(cluster[:,0], cluster[:,1], label='Cluster '+str(i), c=color)
    plt.scatter(centroid[:,0], centroid[:,1], marker='X', label='Centroids')
    plt.legend()
    plt.title('cluster')
    plt.show()

### RESULTS ###
# 01 Plot the data points
plt.scatter(data[:,0], data[:,1], label='data')
plt.title('data point')
plt.legend()
plt.show()

# 02 Visualize the initial condition of the point labels
plot_clusters(labels[0], centroids[0])

# 03 Plot the loss curve
plt.plot(loss)
plt.title('loss')
plt.show()


# 04 Plot the centroid of each cluster
color_it = iter(cm.rainbow(np.linspace(0,1,k)))
for i in range(k):
    color = next(color_it)
    plt.plot(centroid_distances[:,i], label='Cluster '+str(i), c=color)
plt.legend()
plt.title('centroid of cluster')
plt.show()

# 05 Plot the final clustering result
plot_clusters(labels[-1], centroids[-1])