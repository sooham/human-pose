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
iters = 800  # number of iterations
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
# now show the outputs on the test set
for _, image_color, image_bw in test_generator:
    plt.figure(2)
    a = plt.subplot(121)
    a.imshow(image_color.reshape(96, 117, 3))
    a = plt.subplot(122)
    x = np.hstack((image_bw.reshape(1, -1), np.ones((1, 1), dtype=np.float32)))
    result = np.dot(x, W).reshape(96, 117)
    print(result[0, 3:10])
    a.matshow(result)
    plt.show()
    if raw_input('save?') == 'y':
        plt.savefig(input('filename > '))
    plt.close(2)
