import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('profit_population.txt', delimiter=',')
global n
n = data.shape[0]
x_train = data[:,0].reshape([n,1])
y_train = data[:,1].reshape([n,1])


#w = np.zeros(np.shape(data)[1])
def f_pred(X, w):
    insertedX = np.insert(X, 0, 1, axis=1)
    return np.sum(insertedX * w, axis=1).reshape([X.shape[0],1])

def loss(y_pred, y):
    return (np.matmul( (y_pred-y).T, (y_pred-y) )/n/2)[0]

y_pred = f_pred(x_train, np.array([1,1]))

def gradient(y_pred, y, X):
    return (np.matmul(np.ones((1,n)), (y_pred-y)*X)/n)[0]
    #return np.average((y_pred-y)*X, axis=0)

def grad_desc(X, y, w_init, learning_rate, max_iter):
    weights = [w_init]
    w = w_init
    y_pred = f_pred(X, w)
    loss_train = [loss(y_pred, y)]

    for i in range(max_iter):
        grad_f = gradient(y_pred, y, X)

        w = w - learning_rate*grad_f
        y_pred = f_pred(X, w)
        loss_train.append(loss(y_pred, y))

        weights.append(w)
    return w, loss_train, weights

import time
start = time.time()
w_init = np.ones(np.shape(data)[1])
learning_rate = 0.005
max_iter = 10000

w, loss_train, weights = grad_desc(x_train, y_train, w_init, learning_rate, max_iter)
print('Time=',time.time()-start)
for i in range(10):
    print(loss_train[i])


# Plot the training data points
plt.scatter(x_train, y_train)
plt.show()

# Plot the loss curve in the course of gradient descent
plt.plot(loss_train)
plt.show()

# Plot the prediction function superimposed on the training data
plt.scatter(x_train, y_train)
x_range = np.array([min(x_train), max(x_train)])
#plt.plot(x_range, w[0]+x_range*w[1], c='r')
plt.plot(x_range, f_pred(x_range, w), c='r')
plt.show()


#