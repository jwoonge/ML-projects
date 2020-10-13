import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import copy

data_train = np.loadtxt('training.txt', delimiter=',')
data_test = np.loadtxt('testing.txt', delimiter=',')

def sigmoid(z):
    return 1/(1+np.exp(-z + np.exp(-64)))

def vectorize(xs, degrees):
    X = np.ones([xs.shape[0], degrees.shape[0]])
    for i in range(degrees.shape[0]):
        X[:,i] = f(xs, degrees[i])
    return X

def f(xs, degree):
    ret = np.ones((xs.shape[0]))
    for i in range(xs.shape[1]):
        ret *= xs[:,i]**degree[i]
    return ret

def f_pred(X, w):
    return sigmoid(np.matmul(X, w))

def ce_loss(y, y_pred, w, lam):
    return -1*np.average(y*np.log(y_pred + np.exp(-64)) + (1-y)*np.log(1-y_pred + np.exp(-64))) + lam/2*np.average(w)

def grad_ce_loss(X, y, y_pred):
    return 2/X.shape[0] * np.dot(X.T, (y_pred-y))

def accuracy(y, y_pred):
    correct = y_pred[np.abs(y-y_pred)<=0.5]
    return correct.shape[0] / y.shape[0] * 100

def grad_desc(data_train, data_test, degrees, w_init, tau, lam, max_iter):
    X_train = vectorize(data_train[:,:2], degrees)
    X_test = vectorize(data_test[:,:2], degrees)
    label_train = data_train[:,-1].reshape((data_train.shape[0],1))
    label_test = data_test[:,-1].reshape((data_test.shape[0], 1))
    w = copy.deepcopy(w_init)
    y_pred_train = f_pred(X_train, w)
    y_pred_test = f_pred(X_test, w)
    loss_train = [ce_loss(label_train, y_pred_train, w, lam)]
    loss_test = [ce_loss(label_test, y_pred_test, w, lam)]
    acc_train = [accuracy(label_train, y_pred_train)]
    acc_test = [accuracy(label_test, y_pred_test)]
    for i in range(max_iter):
        grad = grad_ce_loss(X_train, label_train, y_pred_train)
        w = (1-tau*lam)*w - tau*grad
        y_pred_train = f_pred(X_train, w)
        y_pred_test = f_pred(X_test, w)
        loss_train.append(ce_loss(label_train, y_pred_train, w, lam))
        loss_test.append(ce_loss(label_test, y_pred_test, w, lam))
        acc_train.append(accuracy(label_train, y_pred_train))
        acc_test.append(accuracy(label_test, y_pred_test))
    return w, loss_train, loss_test, acc_train, acc_test

degrees = np.array([[x,y] for x in range(10) for y in range(10)])
w_init = np.random.randn(100).reshape((100,1))
w_e1, loss_train_e1, loss_test_e1, acc_train_e1, acc_test_e1 = grad_desc(data_train, data_test, degrees, w_init, 0.01, 0.1, 5000)
w_e2, loss_train_e2, loss_test_e2, acc_train_e2, acc_test_e2 = grad_desc(data_train, data_test, degrees, w_init, 0.01, 0.01, 5000)
w_e3, loss_train_e3, loss_test_e3, acc_train_e3, acc_test_e3 = grad_desc(data_train, data_test, degrees, w_init, 0.01, 0.001, 5000)
w_e4, loss_train_e4, loss_test_e4, acc_train_e4, acc_test_e4 = grad_desc(data_train, data_test, degrees, w_init, 0.01, 0.0001, 5000)
w_e5, loss_train_e5, loss_test_e5, acc_train_e5, acc_test_e5 = grad_desc(data_train, data_test, degrees, w_init, 0.01, 0.00001, 5000)


def plot_data(data, title, xmin=-2,xmax=3,ymin=-1,ymax=1.2):
    plt.scatter(data[data[:,2]==0,0],data[data[:,2]==0,1], c='r', s=5)
    plt.scatter(data[data[:,2]==1,0],data[data[:,2]==1,1], c='b', s=5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(['label=0','label=1'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)

def plot_loss_curve(loss_train, loss_test, title):
    plt.plot(loss_train, c='b')
    plt.plot(loss_test, c='r')
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.title(title)
    plt.legend(['training','testing'])


###### OUTPUT ######
# 1. Plot the training data
plot_data(data_train, 'training data')
plt.show()
# 2. Plot the testing data
plot_data(data_test, 'testing data')
plt.show()
# 3. Plot the learning curve with lambda = 0.00001
plot_loss_curve(loss_train_e5, loss_test_e5, title='lambda = 1e-05')
plt.show()
# 4. Plot the learning curve with lambda = 0.0001
plot_loss_curve(loss_train_e4, loss_test_e4, title='lambda = 1e-04')
plt.show()
# 5. Plot the learning curve with lambda = 0.001
plot_loss_curve(loss_train_e3, loss_test_e3, title='lambda = 1e-03')
plt.show()
# 6. Plot the learning curve with lambda = 0.01
plot_loss_curve(loss_train_e2, loss_test_e2, title='lambda = 1e-02')
plt.show()
# 7. Plot the learning curve with lamdba = 0.1
plot_loss_curve(loss_train_e1, loss_test_e1, title='lambda = 1e-01')
plt.show()
# 8. Plot the probability map of obtained classifier with lambda = 0.00001
# 9. Plot the probability map of obtained classifier with lambda = 0.0001
# 10. Plot the probability map of obtained classifier with lambda = 0.001
# 11. Plot the probability map of obtained classifier with lambda = 0.01
# 12. Plot the probability map of obtained classifier with lambda = 0.1

# 13. Plot the final training accuracy with the given regularization parameters
print('lambda\t Training Accuracy (%)')
print('0.00001\t',acc_train_e5[-1])
print('0.0001\t',acc_train_e4[-1])
print('0.001\t',acc_train_e3[-1])
print('0.01\t',acc_train_e2[-1])
print('0.1\t',acc_train_e1[-1])
# 14. Plot the final testing accuracy with the given regularization parameters
print('lambda\t TestingAccuracy (%)')
print('0.00001\t',acc_test_e5[-1])
print('0.0001\t',acc_test_e4[-1])
print('0.001\t',acc_test_e3[-1])
print('0.01\t',acc_test_e2[-1])
print('0.1\t',acc_test_e1[-1])