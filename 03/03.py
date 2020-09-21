import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

data_clean = np.genfromtxt('data_clean.txt', delimiter=',')
data_noisy = np.genfromtxt('data_noisy.txt', delimiter=',')

data_clean = data_clean[:,0:3]
data_noisy = data_noisy[:,0:3]

n_clean = data_clean.shape[0]
n = data_clean.shape[0]
global nd ; nd = 10

def calX(xs, degrees):
    X = np.ones([xs.shape[0], nd])
    for i in range(nd):
        X[:,i] = f(xs, degrees[i])
    return X

def f(xs, degree):
    ret = np.ones((xs.shape[0]))
    for i in range(len(degree)):
        ret *= (xs[:,i]**degree[i])
    return ret

def f_pred(X, w):
    return np.dot(X, w)

def loss(v_pred, v_origin):
    return np.average(np.square(v_pred-v_origin))

def gradient(v_pred, v_origin, X):
    return 2/n * np.dot(X.T, (v_pred-v_origin))

def grad_desc(X, label, w_init, tau, max_iter):
    weights = [w_init]
    w = w_init
    v_pred = f_pred(X, w)
    loss_train = [loss(v_pred, label)]
    for i in range(max_iter):
        grad = gradient(v_pred, label, X)
        w = w - tau*grad
        v_pred = f_pred(X, w)
        loss_train.append(loss(v_pred, label))
        weights.append(w)
    return w, loss_train, np.array(weights)

label = data_noisy[:,-1].reshape((n,1))
degrees = np.array([[0,0], [1,0], [0,1], [1,1], [2,0], [0,2], [2,1], [1,2], [2,2], [4,4]])
w_init = np.ones((10,1))
X = calX(data_noisy, degrees)
tau = 0.1
max_iter = 1000

w, loss_train, weights = grad_desc(X, label, w_init, tau, max_iter)



### Plot the clean data in 3D cartesian coordinate system ###
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data_clean[:,0], data_clean[:,1], data_clean[:,2], s=1)
ax.set_zlim(-1,1)
plt.show()

### Plot the noisy data in 3D cartesian coordinate system ###
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data_noisy[:,0], data_noisy[:,1], data_noisy[:,2], color='r', s=1)
#ax.set_zlim(-1,1)
plt.show()

### Plot the loss curve in the course of gradient descent ###
plt.plot(loss_train)
plt.show()

### Print out the final loss value at convergence of the gradient descent ###
print('loss at convergence = ', loss_train[-1])

### Print out the final model parameter values at convergence of the gradient descent ###
for i in range(10):
    print('model parameter: w_{0} = {1}'.format(i,weights[-1][i]))

### Plot the prediction function in 3D cartesian coordinate system ###
x_coordinate = np.linspace(-1,1,60)
y_coordinate = np.linspace(-1,1,60)
x_pred, y_pred = np.meshgrid(x_coordinate, y_coordinate, indexing='xy')
z_pred = Z = np.zeros((x_coordinate.size, y_coordinate.size))
for (i,j),v in np.ndenumerate(z_pred):
    if i==len(x_coordinate) or j==len(y_coordinate):
        break
    tmp = np.array([x_coordinate[i],y_coordinate[j]]).reshape((1,2))
    tX = calX(tmp, degrees)
    z_pred[j,i] = f_pred(tX, weights[-1])[0][0]
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_surface(x_pred, y_pred, z_pred,rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax.set_zlim(-1,1)
plt.show()


### Plot the prediction functions superimposed on the training data ###
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot_surface(x_pred, y_pred, z_pred,rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax.scatter(data_noisy[:,0], data_noisy[:,1], data_noisy[:,2], color='r', s=1)
ax.set_zlim(-1,1)
plt.show()