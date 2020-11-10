## Principal Component Analysis (PCA) ##
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data-pca.txt', delimiter=',')
#data = np.matmul(data, np.array([[0.85,-0.5],[0.5,0.85]]))

def normalize_data(data):
    mean = np.average(data, axis=0)
    stdev = np.std(data, axis=0)
    return (data-mean)/stdev

data_norm = normalize_data(data)
x_norm = data_norm[:,0]
y_norm = data_norm[:,1]

def covariance(data):
    return np.matmul(data.T, data)/data.shape[0]

covar = covariance(data_norm)

def principal_direction(covariance):
    eig_value, eig_vector = np.linalg.eig(covariance)
    print(eig_value, eig_vector)
    tmp = []
    dim = len(eig_value)
    for i in range(dim):
        tmp.append([eig_value[i], eig_vector[:,i]])
    tmp.sort(key=lambda x:x[0], reverse=True)
    eig_values = []; eig_vectors = []; axis = []
    for i in range(dim):
        eig_value = tmp[i][0]
        eig_vector = tmp[i][1]
        eig_values.append(eig_value)
        eig_vectors.append(eig_vector)
        axis.append(eig_vector*eig_value)
    return np.array(eig_values), np.array(eig_vectors), np.array(axis)

eig_value, eig_vector, axis = principal_direction(covar)

def projection(point, axis):
    ks = np.dot(point, axis)/np.sum(np.square(axis))
    ks = ks.reshape((len(ks),1))
    return np.matmul(ks, axis.reshape(1,len(axis)))

projected1 = projection(data_norm, axis[0,:])
projected2 = projection(data_norm, axis[1,:])
slope1 = axis[0][1]/axis[0][0]
slope2 = axis[1][1]/axis[1][0]

def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1-p2)))



## Results ##
# 1. Plot the origintal data points
plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], c='r', s=3)
plt.title('original data points')
plt.show()

# 2. Plot the normalized data points
arange = np.array([-3,3])
plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3) 
plt.xlim(arange)
plt.ylim(arange)
plt.title('data normalized by z-scoring')
plt.show()

# 3. Plot the principal axis
plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.xlim(arange)
plt.ylim(arange)
plt.plot([0, axis[0][0]], [0, axis[0][1]], c='b')
plt.plot([0, axis[1][0]], [0, axis[1][1]], c='g')
plt.title('principal diretions')
plt.show()

# 4. Plot the first principal axis


plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.xlim(arange)
plt.ylim(arange)
plt.plot(arange,arange*slope1, c='b')
plt.title('first principle axis')
plt.show()

# 5. Plot the project of the normalized data points onto the fist principal axis
plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.xlim(arange)
plt.ylim(arange)
plt.plot(arange, arange*slope1, c='b')
plt.scatter(projected1[:,0], projected1[:,1], c='g')
plt.title('projection to the first principle axis')
plt.show()

# 6. Plot the lines between the normalized data points and their projection points on the fist principal axis
plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.xlim(arange)
plt.ylim(arange)
plt.plot(arange, arange*slope1, c='b')
plt.scatter(projected1[:,0], projected1[:,1], c='g')
for i in range(len(data_norm)):
    plt.plot([projected1[i,0], data_norm[i,0]], [projected1[i,1], data_norm[i,1]])

plt.title('distance to the first principal axis')
plt.show()

# 7. Plot the second principal axis
plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.xlim(arange)
plt.ylim(arange)
plt.plot(arange,arange*slope2, c='b')
plt.title('second principle axis')
plt.show()

# 8. Plot the project of the normalized data points onto the second principal axis
plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.xlim(arange)
plt.ylim(arange)
plt.plot(arange, arange*slope2, c='b')
plt.scatter(projected2[:,0], projected2[:,1], c='g')
plt.title('projection to the second principle axis')
plt.show()


# 9. Plot the lines between the normalized data points and their projection points on the second principal axis
plt.figure(figsize=(6,6))
plt.scatter(x_norm, y_norm, c='r', s=3)
plt.xlim(arange)
plt.ylim(arange)
plt.plot(arange, arange*slope1, c='b')
plt.scatter(projected2[:,0], projected2[:,1], c='g')
for i in range(len(data_norm)):
    plt.plot([projected2[i,0], data_norm[i,0]], [projected2[i,1], data_norm[i,1]])

plt.title('distance to the second principal axis')
plt.show()
