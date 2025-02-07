import os
import pickle
import numpy as np
from utils import kl_mvn




np.random.seed(1)
num_data = 1000
dim = 10

A1_vector = np.random.uniform(0.8, 1.2, dim)
A2_vector = A1_vector
A1 = np.diag(A1_vector)
A2 = np.diag(A2_vector)
B1_bias = np.random.normal(0, 1, dim)
B2_bias = B1_bias

epsilon_list = np.arange(0.0, 0.21, 0.01)

start_index = 0
end_index = 20

PATH = './result/'
if not os.path.isdir(PATH):
    try:
        os.makedirs(PATH)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(PATH):
            pass
        else:
            raise
    
for iter_index in range(start_index, end_index):
    epsilon = epsilon_list[iter_index]
    
    np.random.seed(iter_index)
    A2_vector = A1_vector * (1 + epsilon)
    A1 = np.diag(A1_vector)
    A2 = np.diag(A2_vector)
    B2_bias = B1_bias + epsilon

    mu = B1_bias
    sigma = np.power(A1, 2)
    
    dist = -kl_mvn((mu, sigma), (B1_bias, np.power(A1, 2))) + kl_mvn((mu, sigma), (B2_bias, np.power(A2, 2)))
    print(dist)

    num_seed = 1000

    delta_list = np.zeros([num_seed])
    std_list = np.zeros([num_seed])
    
    for seed in range(num_seed):
        np.random.seed(iter_index * num_seed + seed)
        
        Y1_orig = np.random.normal(0, 1, (num_data, dim)) * A1_vector + B1_bias
        Y2_orig = np.random.normal(0, 1, (num_data, dim)) * A2_vector + B2_bias
        Y_orig = np.random.normal(0, 1, (num_data, dim)) * A1_vector + B1_bias

        g1_inverse = (Y_orig - B1_bias) / A1_vector
        g2_inverse = (Y_orig - B2_bias) / A2_vector
        d1 = dim
        d2 = dim
    
        delta_pp = np.log(np.abs(A2_vector / A1_vector)).sum() -0.5 * np.power(g1_inverse, 2).sum(1).mean() + 0.5 * np.power(g2_inverse, 2).sum(1).mean() + (d2 - d1) * np.log(2 * np.pi)
        std_pp = (-0.5 * np.power(g1_inverse, 2).sum(1) + 0.5 * np.power(g2_inverse, 2).sum(1)).std() / np.sqrt(num_data)

        delta_list[seed] = delta_pp
        std_list[seed] = std_pp
        
        if seed % 50 == 0 or seed == num_seed - 1:
            print("seed: {}".format(seed))    
            filename = "{}/epsilon_{}_ours_result.pt".format(PATH, iter_index)
            f = open(filename, 'wb')
            pickle.dump([delta_list, std_list], f)
            f.close()
    
        
