#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[14]:


def initialize_parameters(layer_dimension):
    parameters = {}
    L = len(layer_dimension)
    for i in range(1,L+1):
        parameters['W' + str(i)] = np.random.randn(layer_dimension[i] , layer_dimension[i-1])
        parameters['b' + str(i)] = np.zeros((layer_dimension[i] , 1))
    return parameters


# In[15]:


def sigmoid(Z):
    a = 1/(1+np.exp(-Z))
    return a


# In[16]:


def relu(Z):
    a = np.max(0,Z)
    return a


# In[18]:


def forward(A_prev,W,b,activation):
    Z = np.dot(W,A_prev) + b
    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)
    cache = (A_prev, W, b, Z, A)
    return A,cache


# In[19]:


def model_forward(X,parameters,activation):
    A = X
    L = len(parameters)/2
    caches = []
    for l in range(1,L+1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A,cache = forward(A,W,b,activation)
        caches.append(cache)
    return A, caches
        


# In[20]:


def cost(Y,h):
    return np.sum(-(Y*np.log(h) + (1-Y)*np.log(1-h)))


# In[ ]:


def activation_derivative(Z, activation):
    if activation == 'sigmoid':
        derivative = sigmoid(Z)*(1 - sigmoid(Z))
    elif activation == 'relu':
        derivative = np.where(Z > 0, 1, 0)
    return derivative


# In[21]:


def backward(dZ_next, W_next, Z, A_prev, activation):
    m = A_prev.shape[1]
    g_derivative = activation_derivative(Z, activation)
    dZ = np.dot(np.transpose(W_next), (dZ_next * g_derivative))
    dW = np.dot(dZ, np.transpose(A_prev))
    db = np.dot(dZ, np.ones(m, 1))
    return (dW, db, dZ)


# In[22]:


def model_backward(Y, Y_predict, caches, activation):
    L = len(caches)
    m = Y.shape[1]
    gradients = {}
    A_prev, W, b, Z, A = caches[L-1]
    g_derivative = activation_derivative(Z, activation)
    dZ = (((1-Y)/(1-Y_predict)) - (Y/Y_predict)) * g_derivative
    gradients['W'+str(L)] = np.dot(dZ, np.transpose(A_prev))
    gradients['b'+str(L)] = np.dot(dZ, np.ones(m, 1))
    for l in range(L-1, 0, -1):
        A, W_next, b_next, Z_next, A_next = A_prev, W, b, Z, A
        A_prev, W, b, Z, A = caches[l-1]
        dW, db, dZ = backward(dZ, W_next, Z, A_prev, activation)
        gradients['W'+str(l)] = dW
        gradients['b'+str(l)] = db
    return gradients


# In[23]:


def update_parameters(parameters, gradients, learning_rate):
    for weight in parameters:
        parameters[weight] = parameters[weight] - learning_rate * gradients[weight]
    return parameters


# In[24]:


def train(X, Y, Layer_dimension, num_iterations, activation, learning_rate):
    parameters = initialize_parameters(layer_dimension)
    for i in range(num_iterations):
        Y_predict,caches = model_forward(X, parameters, activation)
        gradients = model_backward(Y, Y_predict, caches, activation)
        parameters = update_parameters(parameters, gradients, learning_rate)
    return parameters


# In[ ]:




