# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 23:05:55 2017

@author: George
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 23:57:27 2017

@author: George
"""

# GP regression with GPflow


# Imported libraries need for the computations
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.io as sio # This is to import mat. from MATLAB
import GPflow
from matplotlib import pyplot as plt
import time
from sklearn.metrics import mean_squared_error
import random
import statistics

plt.style.use('ggplot')

# In the dataset that I am using, because of the NAN values python strugles to 
# intepret it. So, I do the preprocessing in MATLAB and send the files to Python
mat_contents = sio.loadmat('GPBigDataASAsdr') # Put the mat. file to the working dir
mat_contents = sio.loadmat('GPBigData40k')
mat_contents
X = mat_contents['X']                               # Whole covariable matrix
x = mat_contents['x']                               # Training inputs
y = mat_contents['y']                               # Training outputs
xz = mat_contents['xz']                             # Test inputs
yz = mat_contents['yz']                             # Test outputs

st = time.time()
logt = []
logx = []
logf = []

def logger(x):
    if (logger.i % 10) == 0:
        logx.append(x)
        logf.append(m._objective(x)[0])
        logt.append(time.time() - st)
        #print(logger.i)
    logger.i+=1
logger.i = 1

# Create the GP model
# Composite kernel sum(SE,Noise)
M = 50
M = 100 # Inducing points
M = 250
M = 500
M = 1000
MSE500 = []
timelist500 = []
for i in range(10):
    st = time.time()
    logt = []
    logx = []
    logf = []
    random_n = random.sample(range(len(x)), M)
    Z = x[random_n,:]
    k1 = GPflow.kernels.RBF(input_dim=8, ARD=True)
    k2 = GPflow.kernels.White(input_dim=8)
    k3 = k1 + k2
    m = GPflow.svgp.SVGP(x, y, k3, GPflow.likelihoods.Gaussian(), Z, minibatch_size=5000)
    m.optimize(method=tf.train.AdamOptimizer(0.001), maxiter=20000, callback=logger)
    py, ps2= m.predict_y(xz)
    rMSE = mean_squared_error(yz, py)
    MSE500.append(rMSE)
    timelist500.append(logt[-1])
    print(i)
    print(rMSE)
    print(logt[-1])
    

plt.rcParams['axes.facecolor'] = 'white'
plt.plot(-np.array(logf))
plt.xlabel('Iterations')
plt.ylabel('ELBO')
plt.grid(b=False)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

py, ps2= m.predict_y(xz)
rMSE = mean_squared_error(yz, py)
MSE = []
MSE.append(rMSE)
print(MSE)

# Check the mean MSE and SD(MSE)
statistics.mean(MSE50)
statistics.stdev(MSE50)

print(m.kern.rbf.lengthscales)