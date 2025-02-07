import os
import pickle
import numpy as np
from utils import kl_mvn, kl_divergence_knn, solve_for_B



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
    
    orig_statistic_list = np.zeros([num_seed])
    hulc_statistic_dict = {}

    for seed in range(num_seed):
        np.random.seed(iter_index * num_seed + seed)
        
        Y1_orig = np.random.normal(0, 1, (num_data, dim)) * A1_vector + B1_bias
        Y2_orig = np.random.normal(0, 1, (num_data, dim)) * A2_vector + B2_bias
        Y_orig = np.random.normal(0, 1, (num_data, dim)) * A1_vector + B1_bias

        full_statistics = -kl_divergence_knn(Y_orig, Y1_orig) + kl_divergence_knn(Y_orig, Y2_orig)
        orig_statistic_list[seed] = full_statistics
        data_index = np.arange(num_data)
    
        # Adaptive HulC
        # estimate median bias by subsampling
        num_subsample = 500
        subample_size = int(np.floor(np.sqrt(num_data)))
        subsample_statistic_list = np.zeros([num_subsample])
        for subsample_seed in range(num_subsample):
            subsample_index = np.random.choice(data_index, size=subample_size,  replace = False)
            Y1 = Y1_orig[subsample_index,:]
            Y2 = Y2_orig[subsample_index,:]
            Y = Y_orig[subsample_index,:]
            temp_KL_diff = -kl_divergence_knn(Y, Y1) + kl_divergence_knn(Y, Y2)
            subsample_statistic_list[subsample_seed] = temp_KL_diff
    
        Delta = np.abs((subsample_statistic_list < full_statistics).mean() - 0.5)
        alpha = 0.1
    
        
        B1 = solve_for_B(Delta = Delta)
        B_hulc = B1
        p1 = (1/2 + Delta)**B1 + (1/2 - Delta)**B1
        B0 = B1 - 1
        p0 = (1/2 + Delta)**B0 + (1/2 - Delta)**B0
        U = np.random.uniform(0,1,1)
        tau = (alpha - p1)/(p0 - p1)
        B_hulc = B0*(U <= tau)+ B1*(U > tau)
    
        num_split = int(B_hulc)
        hul_statistic_list = np.zeros([num_split])
        hulc_split_n = num_data // num_split
    
    
        for hulc_seed in range(num_split):
            if hulc_seed == num_split - 1:
                hulc_index = data_index[(hulc_seed * hulc_split_n) : ]
            else:
                hulc_index = data_index[(hulc_seed * hulc_split_n) : ((hulc_seed + 1) * hulc_split_n)]
    
            Y1 = Y1_orig[hulc_index,:]
            Y2 = Y2_orig[hulc_index,:]
            Y = Y_orig[hulc_index,:]
            temp_KL_diff = -kl_divergence_knn(Y, Y1) + kl_divergence_knn(Y, Y2)
            
            hul_statistic_list[hulc_seed] = temp_KL_diff
        hulc_statistic_dict[seed] = hul_statistic_list
    
        if seed % 50 == 0 or seed == num_seed - 1:
            print("seed: {}".format(seed))
    
            filename = "{}/epsilon_{}_KL_hulc_result.pt".format(PATH, iter_index)
            f = open(filename, 'wb')
            pickle.dump([orig_statistic_list, hulc_statistic_dict], f)
            f.close()
    
        
