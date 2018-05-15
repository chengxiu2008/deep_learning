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
  #dev data shape:  (500, 3073)
  #dev labels shape:  (500,)
  #W: (3073, 10)
  num_train = X.shape[0]
  exp_sum = 0
  softmax_dW = np.zeros((num_train, W.shape[1]))
  
  for i in range(num_train):
    scores = X[i].dot(W)  #(10,)    
    max_score =  scores[np.argmax(scores)]
    scores = scores - max_score 
    exp_sum = np.sum(np.exp(scores))
    loss += -np.log(np.exp(scores[y[i]])/exp_sum)
    softmax_dW[i,:] = (np.exp(scores)/exp_sum)    
    softmax_dW[i, y[i]] -= 1 #(500,10)

  dW = X.transpose().dot(softmax_dW)
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  
  dW /= num_train
  dW += reg * W
  
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
  num_train = X.shape[0]
  exp_sum = 0
  softmax_dW = np.zeros((num_train, W.shape[1]))

  score = X.dot(W)  #(500,10)
   
  #nomorlize the value to subtract the maximum in each data
  score -= score[np.arange(num_train), np.argmax(score, axis = 1)][:, np.newaxis]
  exp_sum = np.sum(np.exp(score), axis = 1)
  loss = np.sum(-np.log(np.exp(score[np.arange(num_train),y])/exp_sum))
  softmax_dW = (np.exp(score)/exp_sum[:, np.newaxis])
  softmax_dW[np.arange(num_train), y] -= 1
  dW = X.transpose().dot(softmax_dW)             
                                             
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

