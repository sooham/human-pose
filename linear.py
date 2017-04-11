from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
from autograd import numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp

from utils import PASCAL_context

''' Run a gradient descent linear regression model on the dataset '''
pascal = PASCAL_context()
train, test, labels = pascal.load_dataset(50)
train_generator = train()
test_generator = test()


# define parameters and hyperparameters
iters = 200  # number of iterations
alpha = 0.00    # weight regularization param
learning_rate = 0.0001 # learning rate
print_every = 1

npr.seed(0)
# define model weights
W = np.ones((96*117 + 1, 96*117), dtype=np.float32) * 0.01

def loss(x, t, w):
    ''' returns the loss scalar '''
    z = np.dot(x, w)
    return np.mean(np.sqrt(np.sum(np.square(z - t), axis=1))) + alpha * np.sum(np.square(w))

delta = grad(loss, argnum=2)

# ----------- TRAINING SCHEME --------------
loss_curve = []
for i in range(iters):
    train_size, x, y = train_generator.next()
    # add a bias of ones to x
    x = np.hstack((x, np.ones((x.shape[0], 1), dtype=np.float32)))
    if i % print_every == 0:
        l = loss(x, y, W)
        loss_curve.append(l)
        plt.plot(loss_curve, 'r-')
        plt.pause(0.1)
        plt.close()
        print('%d, loss: %f, weights: %f' % (i, l, np.sum(np.square(W))))
    W -= learning_rate * delta(x, y, W)

print('DONE TRAINING')
np.savez('linear_weights.npz', W=W)

# now show the outputs on the test set and the ground truth
for _, x, y in test_generator:
    plt.figure()
    a = plt.subplot(131)
    a.matshow(x.reshape(96, 117), cmap='gray')
    a.set_xlabel('Input')
    a = plt.subplot(132)
    a.matshow(y.reshape(96, 117))
    a.set_xlabel('Ground Truth')
    a = plt.subplot(133)
    a.set_xlabel('Predicted')
    z = np.hstack((x.reshape(1, -1), np.ones((1, 1), dtype=np.float32)))
    result = np.clip(np.round(np.dot(z, W).reshape(96, 117)), 1, 459)
    a.matshow(result)

    plt.show()

    print(result[0, 3:10])
    print('accuracy: %f' % np.mean(np.equal(result.ravel(), y)))

    r = raw_input('save?')
    if  r == 'y':
        plt.savefig(raw_input('filename > '))
    elif r == 'q':
        break
    plt.close()

m = test()

acc = []
for _, x, y in m:
    z = np.hstack((x.reshape(1, -1), np.ones((1, 1), dtype=np.float32)))
    result = np.clip(np.round(np.dot(z, W).reshape(96, 117)), 1, 459)
    acc.append(np.mean(np.equal(result.ravel(), y)))

print('Total accuracy: %f' % np.mean(acc))