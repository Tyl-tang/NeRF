import matplotlib.pyplot as plt

import numpy as np


def n_gradient(f, x ,w):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    for i in range(2):
        tmp_val = w
        w = tmp_val + h
        fxh1 = f(w)##计算f(x+h)的值

        w = tmp_val - h
        fxh2 = f(w) # f(x-h)
        grad = (fxh1 - fxh2) / (2*h)
        w = tmp_val
    return grad

def n_gradient_2(f, X, w):

    if w.ndim ==1:
        return n_gradient(f,X,w)
    else:
        grad = np.zeros_like(X)
        for idx,x in enumerate(w):
            grad[idx] = n_gradient(f,X,w)
        return grad


def func_2(x):
    return x**2


def MSE(y, t):
    return (y - t) ** 2 / np.size(y)

def gradient_descent(f, init_x, w,lr=0.1, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = n_gradient(f, x,w)
        m = np.sum(grad)/grad.size
        w -= lr * m
        print('w:', w)
    return w


def Relu(x):
    return np.maximum(0, x)

def Network(X,W1,B1,W2,B2):

    Z1= W1*X +B1
    Z_1= Relu(Z1)
    Z2= W2*Z_1 +B2
    Z_2= Relu(Z2)
    RESULT = np.sum(Z_2, axis=0)
    return RESULT

def loss(x, w1,b1,w2,b2):
    t=func_2(x)
    y=Network(x,w1,b1,w2,b2)
    loss = MSE(y,t)
    return loss

def f(W):
    return loss(X, W, b1, w2, b2)

X = np.arange(-2,2,0.1)
w1 = np.random.rand(3,1)
w2 = np.random.rand(3,1)
b1 =np.zeros_like(w1)
b2 =np.zeros_like(w2)
M = Network(X,w1,b1,w2,b2)
l = loss(X,w1,b1,w2,b2)

dW1 = n_gradient(f,X, w1)
#dw2 = n_gradient(f,X, w2)


gradient_descent(f, X, w1,lr=0.1, step_num=100)
