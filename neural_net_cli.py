#!/usr/bin/env python
# coding: utf-8
import numpy as np

def sigmoid(Z):
    a = 1/(1+np.exp(-Z))
    return a

def relu(Z):
    a = np.max(0,Z)
    return a
    
def activation_derivative(Z, activation='sigmoid'):
    if activation == 'sigmoid':
        derivative = sigmoid(Z)*(1 - sigmoid(Z))
    elif activation == 'relu':
        derivative = np.where(Z > 0, 1, 0)
    return derivative

class Model:
    def __init__(self, feature_dimension, activation='sigmoid'):
        self.feature_dimension = feature_dimension
        L = len(self.feature_dimension) - 1
        self.parameters = {}
        self.gradients = {}
        for i in range(1,L+1):
            self.parameters['W' + str(i)] = np.random.randn(self.feature_dimension[i] , self.feature_dimension[i-1])
            self.parameters['b' + str(i)] = np.zeros((self.feature_dimension[i] , 1))
        self.activation = activation
        
    def reset(self):
        L = len(self.feature_dimension) - 1
        
        for i in range(1,L+1):
            self.parameters['W' + str(i)] = np.random.randn(self.feature_dimension[i] , self.feature_dimension[i-1])
            self.parameters['b' + str(i)] = np.zeros((self.feature_dimension[i] , 1))
    
    def forward(self, A_prev, W, b):
        Z = np.dot(W,A_prev) + b
        if self.activation == 'sigmoid':
            A = sigmoid(Z)
        elif self.activation == 'relu':
            A = relu(Z)
        cache = (A_prev, W, b, Z, A)
        return A,cache
    
    def model_forward(self, X):
        A = X
        L = len(self.parameters)/2
        caches = []
        for l in range(1,L+1):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            A,cache = self.forward(A,W,b)
            caches.append(cache)
        return A, caches
    
    def cost(self, Y, h):
        return np.sum(-(Y*np.log(h) + (1-Y)*np.log(1-h)))
    
    def backward(self, dZ_next, W_next, Z, A_prev):
        m = A_prev.shape[1]
        g_derivative = activation_derivative(Z, self.activation)
        dZ = np.dot(np.transpose(W_next), (dZ_next * g_derivative))
        dW = np.dot(dZ, np.transpose(A_prev))
        db = np.dot(dZ, np.ones(m, 1))
        return (dW, db, dZ)
    
    def model_backward(self, Y, Y_predict, caches):
        L = len(caches)
        m = Y.shape[1]
        A_prev, W, b, Z, A = caches[L-1]
        g_derivative = activation_derivative(Z, self.activation)
        dZ = (((1-Y)/(1-Y_predict)) - (Y/Y_predict)) * g_derivative
        self.gradients['W'+str(L)] = np.dot(dZ, np.transpose(A_prev))
        self.gradients['b'+str(L)] = np.dot(dZ, np.ones(m, 1))
        for l in range(L-1, 0, -1):
            A, W_next, b_next, Z_next, A_next = A_prev, W, b, Z, A
            A_prev, W, b, Z, A = caches[l-1]
            dW, db, dZ = self.backward(dZ, W_next, Z, A_prev)
            self.gradients['W'+str(l)] = dW
            self.gradients['b'+str(l)] = db
    
    def update_parameters(self, learning_rate):
        for weight in self.parameters:
            self.parameters[weight] = self.parameters[weight] - learning_rate * self.gradients[weight]
    
    def train(self, X, Y, num_iterations, learning_rate=0.01):
        for i in range(num_iterations):
            Y_predict,caches = self.model_forward(X)
            self.model_backward(Y, Y_predict, caches)
            self.update_parameters(learning_rate)