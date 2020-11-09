## Principal Component Analysis (PCA) ##
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data-pca.txt', delimiter=',')
x = data[:,0]
y = data[:,1]

def normalize_data(data):
    mean = np.average(data, axis=0)
    stdev = np.std(data, axis=0)
    return (data-mean)/stdev

data_norm = normalize_data(data)
x_norm = data_norm[:,0]
y_norm = data_norm[:,1]

def compute_covariance(data):
    return np.matmul(data.T, data)/data.shape[0]

covar = compute_covariance(data)

def compute_principal_direction(covariance):
    eig_value, eig_vector = np.linalg.eig(covariance)
    print(eig_value, eig_vector)
    tmp = []
    dim = len(eig_value)
    for i in range(dim):
        tmp.append([eig_value[i], eig_vector[i]])
    tmp.sort(key=lambda x:x[0], reverse=True)
    eig_value = []; eig_vector = []
    for i in range(dim):
        eig_value.append(tmp[i][0])
        eig_vector.append(tmp[i][1])
    return np.array(eig_value), np.array(eig_vector)

a, b = compute_principal_direction(covar)


def compute_projection(point, axis):
    return None
def compute_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1-p2)))



## Results ##
# 1. Plot the origintal data points
plt.scatter(x, y, c='r', s=3)
plt.show()

# 2. Plot the normalized data points
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.show()

# 3. Plot the principal axis

# 4. Plot the first principal axis

# 5. Plot the project of the normalized data points onto the fist principal axis

# 6. Plot the lines between the normalized data points and their projection points on the fist principal axis

# 7. Plot the second principal axis

# 8. Plot the project of the normalized data points onto the second principal axis

# 9. Plot the lines between the normalized data points and their projection points on the second principal axis