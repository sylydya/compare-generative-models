import os
import pickle
import numpy as np
from utils import wasserstein_gaussian_square, W2_diff

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
    
    dist = -wasserstein_gaussian_square(mu, sigma, B1_bias, np.power(A1, 2)) + wasserstein_gaussian_square(mu, sigma, B2_bias, np.power(A2, 2))
    print(dist)
    
    num_seed = 1000
    num_subsample = 500
    
    orig_statistic_list = np.zeros([num_seed])
    subsample_statistic_list = np.zeros([num_seed, num_subsample])
    
    for seed in range(num_seed):
        np.random.seed(iter_index * num_seed + seed)

        Y1_orig = np.random.normal(0, 1, (num_data, dim)) * A1_vector + B1_bias
        Y2_orig = np.random.normal(0, 1, (num_data, dim)) * A2_vector + B2_bias
        Y_orig = np.random.normal(0, 1, (num_data, dim)) * A1_vector + B1_bias
        
        temp_W2_diff = W2_diff(Y_orig, Y1_orig, Y2_orig)
    
        orig_statistic_list[seed] = temp_W2_diff
        
        data_index = np.arange(num_data)
        subample_size = int(np.floor(np.sqrt(num_data)))
        
        for subsample_seed in range(num_subsample):
            subsample_index = np.random.choice(data_index, size=subample_size, replace = False)
            Y1 = Y1_orig[subsample_index,:]
            Y2 = Y2_orig[subsample_index,:]
            Y = Y_orig[subsample_index,:]
        
            temp_W2_diff = W2_diff(Y, Y1, Y2)
            subsample_statistic_list[seed, subsample_seed] = temp_W2_diff
            
        if seed % 50 == 0 or seed == num_seed - 1:
            print("seed: {}".format(seed))
    
            filename = "{}/epsilon_{}_subsample_result.pt".format(PATH, iter_index)
            f = open(filename, 'wb')
            pickle.dump([orig_statistic_list, subsample_statistic_list], f)
            f.close()

    
