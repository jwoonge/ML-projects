import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# 1. Load dataset
data = np.loadtxt('dataset-a.txt', delimiter=',')
global n; n = data.shape[0]
global nx; nx = data.shape[1]-1

# 2. Define a logistic regression loss function and its gradient
def sigmoid(z):
    return 1/(1+np.exp(-z))

def vectorize(xs, degrees):
    X = np.ones([xs.shape[0], degrees.shape[0]])
    for i in range(nd):
        X[:,i] = f(xs, degrees[i])
    return X

def f(xs, degree):
    ret = np.ones((xs.shape[0]))
    for i in range(nx):
        ret *= (xs[:,i]**degree[i])
    return ret

def f_pred(X, w):
    return sigmoid(np.matmul(X,w))

def ce_loss(y, y_pred):
    return -1*np.average(y*np.log(y_pred + np.exp(-64)) + (1-y)*np.log(1-y_pred + np.exp(-64)))

def grad_ce_loss(X, y, y_pred):
    return 2/n * np.dot(X.T, (y_pred - y))

def accuracy(y, y_pred):
    correct = y_pred[np.abs(y-y_pred)<=0.5]
    return correct.shape[0] / y.shape[0] * 100

# 3. Define a prediction function and run a gradient descent algorithm
def grad_desc(X, y, w_init, tau, max_iter):
    w = w_init
    y_pred = f_pred(X, w)
    loss = [ce_loss(y, y_pred)]
    accuracy_train = [accuracy(y, y_pred)]
    for i in range(max_iter):
        grad = grad_ce_loss(X, y, y_pred)
        w = w - tau*grad
        y_pred = f_pred(X, w)
        loss.append(ce_loss(y, y_pred))
        accuracy_train.append(accuracy(y, y_pred))
    return w, loss, accuracy_train

label = data[:,-1].reshape((n,1))
degrees = np.array([[0,0],[1,0],[0,1],[2,0],[0,2],[2,1],[1,2],[2,2],[1,3],[3,1]])
nd = degrees.shape[0]
w_init = np.ones((nd,1))
X = vectorize(data, degrees)
tau = 0.01; max_iter=10000
w, loss_train, accuracy_train = grad_desc(X, label, w_init, tau, max_iter)

def decision_boundary(w, degrees, minx, maxx, miny, maxy):
    xx1, xx2 = np.meshgrid(np.linspace(minx, maxx), np.linspace(miny, maxy)) # create meshgrid
    X2 = np.ones([np.prod(xx1.shape),2]) 
    X2[:,0] = xx1.reshape(-1)
    X2[:,1] = xx2.reshape(-1)
    p = f_pred(vectorize(X2, degrees),w)
    p = p.reshape([xx1.shape[0],xx2.shape[0]])

    idx_class0 = (data[:,2]==0)
    idx_class1 = (data[:,2]==1)
    plt.scatter(data[idx_class0,0], data[idx_class0,1], s=5, c='r', label='class=0')
    plt.scatter(data[idx_class1,0], data[idx_class1,1], s=5, c='b', label='class=1')
    plt.contour(xx1, xx2, p, levels=[0,0.5,1])
    plt.legend()
    plt.title('Decision Boundary')
    plt.show()

def probability_map(w, degrees, minx, maxx, miny, maxy):
    num_a = 100
    grid_x1 = np.linspace(minx,maxx,num_a); grid_x2 = np.linspace(miny,maxy,num_a)
    score_x1, score_x2 = np.meshgrid(grid_x1, grid_x2)
    Z = np.zeros((len(grid_x1), len(grid_x2)))
    for i in range(len(grid_x1)):
        for j in range(len(grid_x2)):
            tmpX = np.array([grid_x1[i], grid_x2[j]]).reshape((1,nx))
            predict_prob = f_pred(vectorize(tmpX, degrees), w)
            Z[j, i] = predict_prob

    cf = plt.contourf(score_x1, score_x2, Z, num_a, alpha=0.5, cmap='RdBu')
    cbar = plt.colorbar(cf)
    cbar.update_ticks()
    idx_class0 = (data[:,2]==0)
    idx_class1 = (data[:,2]==1)
    plt.scatter(data[idx_class0,0], data[idx_class0,1], s=5, c='r', label='class=0')
    plt.scatter(data[idx_class1,0], data[idx_class1,1], s=5, c='b', label='class=1')
    plt.legend()
    plt.title('Probability Map')
    plt.show()

###### RESULTS ######
# 01 Visualize the data
idx_class0 = (data[:,2]==0)
idx_class1 = (data[:,2]==1)
plt.scatter(data[idx_class0,0], data[idx_class0,1], s=5, c='r')
plt.scatter(data[idx_class1,0], data[idx_class1,1], s=5, c='b')
plt.legend(['class=0','class=1'])
plt.title('Training data')
plt.show()
# 02 Plot the loss curve obtained by the gradient descent
plt.plot(loss_train)
plt.show()
# 03 Plot the decision boundary of the obtained classifier
minx, maxx = data[:,0].min(), data[:,0].max()
miny, maxy = data[:,1].min(), data[:,1].max()
decision_boundary(w, degrees, minx, maxx, miny, maxy)
# 04 Plot the probability map of the obtained classifier
probability_map(w, degrees, minx, maxx, miny, maxy)
# 05 Compute the classification accuracy
print(round(accuracy_train[-1],2))