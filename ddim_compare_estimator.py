import argparse

import os
import time
import torch
import pickle
import numbers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from tqdm import tqdm
# from diffusers import DDIMPipeline
from pytorch_diffusion import Diffusion
from torch.autograd.functional import jacobian


data_name = 'cifar10'
# dog
label_index = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

scratct_path = '.'
save_result_path = "{}/result/{}".format(scratct_path, data_name)

model_name_1 = "ema_{}".format(data_name)
model_name_2 = "{}".format(data_name)

num_ddim_steps = 20
num_diffusion_timesteps = 1000
timesteps = torch.ceil(torch.linspace(0, num_diffusion_timesteps - 1, num_ddim_steps + 1)).long().to(device)
time_range = timesteps

# load image index for a specific label
filename = '{}/label_index_{}_image_index_list.pt'.format(save_result_path, label_index)
f = open(filename, 'rb')
image_index_list = pickle.load(f)
f.close()


estimator_list = []
for image_index in image_index_list:
    image_path = '{}/image_{:d}'.format(save_result_path, image_index)
    # load jacobian matrix and compute the log determinant
    filename = '{}/jac_logdet_all.pt'.format(image_path)
    if os.path.exists(filename):
        filename = '{}/jac_logdet_all.pt'.format(image_path)
        f = open(filename, 'rb')
        [jac_logdet_model1, jac_logdet_model2] = pickle.load(f)
        f.close()
    else:
        # model 1
        product_jac = None
        for i, t in enumerate(time_range[0:-1]):
            filename = '{}/model_{}_time_{:d}.pt'.format(image_path, model_name_1, t)
            f = open(filename, 'rb')
            temp_jac = pickle.load(f)
            f.close()
            if product_jac is None:
                product_jac = temp_jac
            else:
                product_jac = product_jac.matmul(temp_jac)
        
        if torch.det(product_jac) < 0:
            temp = torch.clone(product_jac)
            temp[0,:] = -1 * temp[0,:]
            jac_logdet_model1 = torch.logdet(temp)
        else:
            jac_logdet_model1 = torch.logdet(product_jac)
    
        # model 2
        product_jac = None
        for i, t in enumerate(time_range[0:-1]):
            filename = '{}/model_{}_time_{:d}.pt'.format(image_path, model_name_2, t)
            f = open(filename, 'rb')
            temp_jac = pickle.load(f)
            f.close()
            if product_jac is None:
                product_jac = temp_jac
            else:
                product_jac = product_jac.matmul(temp_jac)
        if torch.det(product_jac) < 0:
            temp = torch.clone(product_jac)
            temp[0,:] = -1 * temp[0,:]
            jac_logdet_model2 = torch.logdet(temp)
        else:
            jac_logdet_model2 = torch.logdet(product_jac)
    
        filename = '{}/jac_logdet_all.pt'.format(image_path)
        f = open(filename, 'wb')
        pickle.dump([jac_logdet_model1, jac_logdet_model2], f)
        f.close()

    # load latent noise
    filename = '{}/model_{}_intermediates.pt'.format(image_path, model_name_1)
    f = open(filename, 'rb')
    intermediates1 = pickle.load(f)
    f.close()
    
    filename = '{}/model_{}_intermediates.pt'.format(image_path, model_name_2)
    f = open(filename, 'rb')
    intermediates2 = pickle.load(f)
    f.close()

    # compute estimator
    estimator = 0.5 * intermediates2[-1].pow(2).sum() - 0.5 * intermediates1[-1].pow(2).sum() + jac_logdet_model1 - jac_logdet_model2
    print("image index: {}, estimator: {}".format(image_index, estimator.item()))
    estimator_list.append(estimator.item())

    filename = '{}/label_index_{}_estimator_list.pt'.format(save_result_path, label_index)
    f = open(filename, 'wb')
    pickle.dump(estimator_list, f)
    f.close()

filename = '{}/label_index_{}_estimator_list.pt'.format(save_result_path, label_index)
f = open(filename, 'wb')
pickle.dump(estimator_list, f)
f.close()

