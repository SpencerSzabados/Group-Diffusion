"""
    Date: 2023-12-24 

    File contains scripts for generating C4 invariant guassian mixture dataset used for test
    model distribution matching. Specifically sampling from a guassian mixture distribution 
    with four equaly spaced modes each a isometric guassian. The sampling method is based on
    that described in: 
    (https://medium.com/analytics-vidhya/sampling-from-gaussian-mixture-models-f1ab9cac2721)
     
    This file also encludes scripts for embedding points data into a square "image" format the
    diffusion model can operate on, and vice versa given the model output. In particular, 2D 
    point data is encoded into the first two channels of a 1x1 image with that last channel
    set to a constant value.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Model parameters
num_samples = 20000
mus = [np.array([0, 1]), np.array([7, 1]), np.array([-3, 3])]
covs = [np.array([[1, 2], [2, 1]]), np.array([[1, 0], [0, 1]]), np.array([[10, 1], [1, 0.3]])]
pis = np.array([0.3, 0.1, 0.6])
acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis)+1)] 
assert np.isclose(acc_pis[-1], 1)

def inv_sigmoid(values):    
    return np.log(values/(1-values))

samples_x = np.zeros(num_samples)
samples_y = np.zeros_like(samples_x)
for j in range(num_samples):
    # sample uniform
    r = np.random.uniform(0, 1)
    # select Gaussian
    k = 0
    for i, threshold in enumerate(acc_pis):    
        if r < threshold:        
            k = i        
            break
    selected_mu = mus[k]
    selected_cov = covs[k]

    # sample from selected Gaussian
    lambda_, gamma_ = np.linalg.eig(selected_cov)
    dimensions = len(lambda_) 
    # sampling from normal distribution
    y_s = np.random.uniform(0, 1, size=(dimensions*1, 3))
    x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
    # transforming into multivariate distribution
    x_multi = (x_normal*lambda_) @ gamma_ + selected_mu
    samples_x[j] = x_multi[0][0]
    samples_y[j] = x_multi[0][1]

plt.scatter(samples_x, samples_y)
plt.show()