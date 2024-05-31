from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]
    D = X.shape[1]

    for i in range(N):
      
      scores = np.zeros(C)

      for j in range(C):
        scores[j] += np.sum( X[i,:] * W[:,j] )

      # Para evitar inestabilidad numérica
      scores -= np.max(scores)

      probs = np.exp(scores)
      probs = probs / np.sum(probs)

      for j in range(C): 
        dW[:, j] += (probs[j] - (j==y[i])) * X[i] 

      loss += -np.log(probs[y[i]])

      

    loss = loss / N
    dW = dW / N

    # Regularización
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

    scores = X @ W

    probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis= 1).reshape(-1,1)

    probs_loss = probs[np.arange(N), y].reshape(-1,1)


    loss = -np.mean(np.log(probs_loss)) + reg * np.sum(W**2)

    probs[np.arange(N), y] -= 1
    dW = X.T @ probs / N + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
