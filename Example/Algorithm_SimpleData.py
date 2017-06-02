""" This module implements 9 functions. The cust_func calculates the objective value
of the cost function. The norm_func calculates the norm value. The computegrad function
calculates gradient of cost funcation given beta values. The backtracking function
returns step size by implementing backtracking rule. The funcation graddescent that implements 
the gradient descent algorithm with the backtracking rule to tune the step-size. The function 
fastgradalgo that implements the fast gradient algorithm. The objective_plot function 
plots the curve of the objective values for both algorithms versus the iteration counter t.
The compute_misclassification_error function calculates the misclassification error. And
the plot_misclassification_error function will plot the curve of misclassification error 
values for both algorithms versus the iteration counter t.
    This module runs in the following order: Import libraries -> Load Data and split to training
and test set -> Find optimal lambda value to optimize the algorithm -> Calculate beta values 
from both gradient algorithm and fast algorithm -> Plot objective value and missclassification error
versus iteration for both algorithms.
"""


# Import Libraries
import pandas as pd
import numpy as np
import sklearn
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import sparse
import matplotlib.pyplot as plt
from multiprocessing import Process
from sklearn.datasets import load_iris

# Load iris data
iris = load_iris()
X, y = iris.data[:100], iris.target[:100]
n,k = iris.data.shape #number of samples and dimension of each sample
k = k+1


# Standardize Data and Divide the data into train , test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.25)
scaler = sklearn.preprocessing.StandardScaler () 
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train[y_train==0] = -1 #Convertto+/-1
y_train[y_train==1] = 1
y_test[y_test==0] = -1
y_test[y_test==1] = 1

#Object Function 
def cost_func(beta, lam, X_train, y_train): #computes cost given predicted and actual values
    n = len(X_train) #number of training examples
    J = (1./n) *sum(np.log(1+ np.exp(-y_train*X_train.dot(beta))))+ lam*beta.dot(beta) 
    return J
def norm(beta):
    return(np.sum(beta**2))

##Computegrad that computes and returns gradient of cost funcation given beta values.
def computegrad(beta, lam, X_trian, y_train):
    n = len(X_train)
    sigmoid = 1/(1+np.exp(y_train*X_train.dot(beta)))
    grad = -(X_train.T.dot(y_train*sigmoid))
    grad = grad/n + lam*2*beta
    return(grad)

## Backtracking Rule
def backtracking (beta,lam, t, max_iter, X_train, y_train):
    grad_x = computegrad(beta, lam, X_train, y_train)
    norm_grad_x = norm(grad_x)  
    found_t = 0
    iter = 0
    while found_t ==0 and iter < max_iter:
        if (cost_func(beta-t*grad_x, lam, X_train, y_train) <= cost_func(beta, lam, X_train, y_train) - 0.5*t*norm_grad_x**2):
            found_t = 1
        elif iter >= max_iter:
            t = t
        else:
            t = t*0.8
            iter = iter+1
    return(t)

"""The funcation graddescent that implements the gradient descent algorithm 
with the backtracking rule to tune the step-size. The function graddescent calls 
computegrad and backtracking as subroutines. The function takes as input the initial point, 
the initial step-size value, and the maximum number of iterations. 
The stopping criterion is the maximum number of iterations."""
def graddescent (beta, lam, step_init, max_iter=1000):
    x = beta
    grad_x = computegrad(x, lam, X_train, y_train)
    x_vals = [x]
    iter =1
    while iter < max_iter:
        t = backtracking(x,lam, step_init, max_iter, X_train, y_train)
        x = x-t*grad_x
        x_vals.append(x)
        grad_x = computegrad(x, lam, X_train, y_train)
        iter = iter +1
    return(x_vals)

"""The function fastgradalgo that implements the fast gradient algorithm. The function fastgradalgo 
calls computegrad and backtracking as subroutines. 
The function takes as input the initial step-size value for the backtracking rule 
and the maximum number of iterations. The stopping criterion is the maximum number of iterations."""
def fastgradalgo(beta_init, theta_init,lam, step_init, max_iter):
    x = beta_init
    theta = theta_init
    x_vals =[x]
    iter = 1
    while iter < max_iter:   
        t = backtracking(x, lam,step_init, max_iter, X_train, y_train)
        grad_x = computegrad(theta, lam, X_train, y_train)     
        x = theta - t*grad_x
        x_vals.append(x)
        theta = x_vals[iter] + (iter/(iter+3))*(x_vals[iter] - x_vals[iter-1])
        iter = iter + 1
    return(x_vals)

##Plot the curve of the objective values for both algorithms versus the iteration counter t
def objective_plot (beta_vals, beta_vals_fast,lam): 
    grad_vals = [cost_func(i, lam, X_train, y_train) for i in beta_vals]
    grad_vals_fast = [cost_func(i, lam, X_train, y_train) for i in beta_vals_fast]
    fig, ax = plt.subplots()
    ax.plot(range(max_iter), grad_vals, 'r', label='gradient descent')
    ax.plot(range(max_iter), grad_vals_fast, 'b', label='fast gradient descent')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective value vs. iteration when lambda='+str(lam))
    ax.legend(loc='upper right')
    plt.show()

## misclassification error formula
def compute_misclassification_error(beta_opt, x, y):
    y_pred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1
    return np.mean(y_pred != y)

##Plot the curve of the misclassfication error values for both algorithms versus the iteration counter t
def plot_misclassification_error(betas_grad, betas_fastgrad, x, y,lam_opt, title=''):
    niter = np.size(betas_grad, 0)
    error_grad = np.zeros(niter)
    error_fastgrad = np.zeros(niter)
    for i in range(niter):
        error_grad[i] = compute_misclassification_error(betas_grad[i], x, y)
        error_fastgrad[i] = compute_misclassification_error(betas_fastgrad[i], x, y)
    fig, ax = plt.subplots()
    ax.plot(range(1, niter + 1), error_grad, 'r', label='gradient descent')
    ax.plot(range(1, niter + 1), error_fastgrad, 'b',label='fast gradient')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
    plt.title(title)
    ax.legend(loc='upper right')
    plt.show()
     
# Run cross-validation using scikit-learn to find the optimal value of lambda
param_grid = {'C': np.logspace(-10, 10, 100) }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
clf.fit(X_train, y_train)        
lam_opt = 1/(clf.best_estimator_.C) 

"""Run previous functions to return the optimal lambda value, boptimal beta values, and objective
value vs. iteration plot and misclassification error vs. iteration plot by initilizaing initial
beta values and maximum iterations."""
beta_init = np.zeros(k-1) 
theta_init = np.zeros(k-1) 
max_iter=1000
beta_vals2 = graddescent(beta_init,lam_opt, step_init=0.1, max_iter=1000)
beta_vals_fast2 = fastgradalgo(beta_init, theta_init, lam_opt,step_init=0.1, max_iter=1000)
beta_opt = beta_vals2[max_iter-1]
print( 'Optimal lambda=' , lam_opt)
print("beta calculated from fast gradient algorithm\n{}".format(beta_opt))
print("objective value:\n{}".format(cost_func(beta_opt, lam_opt, X_train, y_train)))
objective_plot(beta_vals2, beta_vals_fast2,lam_opt)
plot_misclassification_error(beta_vals2,beta_vals_fast2, X_train, y_train,lam_opt, title='Training Set misclassification error')
plot_misclassification_error(beta_vals2,beta_vals_fast2, X_test, y_test,lam_opt, title ='Test set misclassification error')
