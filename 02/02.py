import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import time

data = np.loadtxt('profit_population.txt', delimiter=',')
global n
n = data.shape[0]
x_train = data[:,0].reshape([n,1])
y_train = data[:,1].reshape([n,1])
X = np.insert(x_train, 0, 1, axis=1)

def f_pred(X, w):
    return np.dot(X,w)

def loss(y_pred, y):
    return (np.matmul( (y_pred-y).T, (y_pred-y) )/n/2)[0][0]

def gradient(y_pred, y, X):
    return 2/n * np.dot(X.T, (y_pred-y))

def grad_desc(X, y, w_init, learning_rate, max_iter):
    weights = [w_init]
    w = w_init
    y_pred = f_pred(X, w)
    loss_train = [loss(y_pred, y)]
    for i in range(max_iter):
        grad = gradient(y_pred, y, X)
        w = w - learning_rate*grad
        y_pred = f_pred(X, w)
        loss_train.append(loss(y_pred, y))
        weights.append(w)
    return w, loss_train, np.array(weights)

### Linear Regression with gradient descent ###
start = time.time()
w_init = np.ones((np.shape(data)[1],1))
learning_rate = 0.01
max_iter = 1000

w, loss_train, weights = grad_desc(X, y_train, w_init, learning_rate, max_iter)
print('Time=',time.time()-start)
for i in range(10):
    print(loss_train[i])
### Linear Regression with Scikit-learn ###
start = time.time()
lin_reg_sklearn = LinearRegression()
lin_reg_sklearn.fit(x_train, y_train)
print('Time=',time.time()-start)

w_sklearn = np.array([lin_reg_sklearn.intercept_, lin_reg_sklearn.coef_[0]])
y_pred_sklearn = f_pred(X, w)
loss_sklearn = loss(y_pred_sklearn, y_train)


# Plot the training data points
plt.scatter(x_train, y_train)
plt.show()

# Plot the loss curve in the course of gradient descent
plt.plot(loss_train)
plt.show()

# Plot the prediction function superimposed on the training data
plt.scatter(x_train, y_train)
x_range = np.array([min(x_train), max(x_train)])
plt.plot(x_range, w[0]+w[1]*x_range, c='r')
plt.show()

# Plot the prediction functions by both the Scikit-learn and the gradient descent
plt.scatter(x_train, y_train)
x_range = np.array([min(x_train), max(x_train)])
plt.plot(x_range, w[0]+w[1]*x_range, c='r')
plt.plot(x_range, w_sklearn[0]+w_sklearn[1]*x_range, c='g')
plt.legend(['gradient descent', 'scikit-learn'])
plt.show()


# Plot the loss surface
B0 = np.linspace(-10, 10, 50)
B1 = np.linspace(-1, 4, 50)
xx, yy = np.meshgrid(B0, B1, indexing='xy')

Z = np.zeros((B0.size, B1.size))

for (i,j),v in np.ndenumerate(Z):
    y_pred = f_pred(X, [[B0[i]],[B1[j]]])
    Z[i,j] = loss(y_pred, y_train)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx,yy,Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('loss')

theta0 = list(weights[:,0,0])
theta1 = list(weights[:,1,0])

ax.plot(theta0, theta1, np.array(loss_train), c='k')
plt.show()

#CS = 
#plt.show()

# Plot the contour on the loss surface
plt.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap = plt.cm.jet)
plt.plot(theta0, theta1, c='k', zorder=5)
plt.scatter(theta0[-1], theta1[-1], c='r')
plt.show()