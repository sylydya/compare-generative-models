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


parser = argparse.ArgumentParser(description='cifar10')
# specify the image label, default 5 (dog)
parser.add_argument('--label_index', default=5, type = int, help = 'image label index')
# specify the image intex range to compute the jacobian
parser.add_argument('--lb', default=0, type = int, help = 'image index lower bound')
parser.add_argument('--ub', default=100, type = int, help = 'image index upper bound')
args = parser.parse_args()

image_label_index = args.label_index
image_index_lower_bound = args.lb
image_index_upper_bound = args.ub


# path to save jacobian reesults
data_name = 'cifar10'
scratct_path = '.'
save_result_path = "{}/result/{}".format(scratct_path, data_name)
if not os.path.isdir(save_result_path):
    try:
        os.makedirs(save_result_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(PATH):
            pass
        else:
            raise

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
    return a.item()

def diffusion_step(img, model, t, alpha, alpha_next, device):
    b = img.shape[0]
    ts = torch.full((b,), t, dtype=torch.long).to(device)
    model.eval()
    e_t = model(img, ts)

    a_t = torch.full((b, 1, 1, 1), alpha, device=device)
    a_next = torch.full((b, 1, 1, 1), alpha_next, device=device)
    sqrt_one_minus_at = (1 - a_t).sqrt()

    # current prediction for x_0
    pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_xt = (1. - a_next).sqrt() * e_t
    x_prev = a_next.sqrt() * pred_x0 + dir_xt
    return x_prev

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# CIFAR10 Data
data_path = "~/workspace/compare-generative-models/data/"
test_transform = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor()]
        )
test_dataset = CIFAR10(
            data_path,
            train=False,
            download=True,
            transform=test_transform,
        )
test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

# DDIM Setting
num_ddim_steps = 20
num_diffusion_timesteps = 1000
timesteps = torch.ceil(torch.linspace(0, num_diffusion_timesteps - 1, num_ddim_steps + 1)).long().to(device)
beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps
        )
betas = torch.from_numpy(betas).float().to(device)


# load pretrained model, ema_cifar10 and cifar10
model_name_1 = "ema_{}".format(data_name)
diffusion_model_1 = Diffusion.from_pretrained(model_name_1)
model_1 = diffusion_model_1.model
model_1.to(device)

model_name_2 = "{}".format(data_name)
diffusion_model_2 = Diffusion.from_pretrained(model_name_2)
model_2 = diffusion_model_2.model
model_2.to(device)

# load image with specific labels
for image_index, (x, y) in enumerate(test_loader):
    if image_index < image_index_lower_bound:
        continue
    elif image_index >= image_index_upper_bound:
        break
    elif y != image_label_index:
        continue
    else:
        img = 2 * x - 1
        print(image_index, y)
        img = img.to(device)
        x0 = img
        image_path = '{}/image_{:d}'.format(save_result_path, image_index)
        if not os.path.isdir(image_path):
            try:
                os.makedirs(image_path)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise

        # forward pass for model 1
        img = x0
        log_every_t = 1
        intermediates = [x0]
        time_range = timesteps
        iterator = tqdm(time_range[0:-1], desc='DDIM Forward', total=num_ddim_steps)
        
        for i, t in enumerate(iterator):
            # compute stepwise jacobian and save the result
            alpha = compute_alpha(betas, time_range[i])
            alpha_next = compute_alpha(betas, time_range[i + 1])
            def jac_encode(img):
                return diffusion_step(img, model_1, t, alpha, alpha_next, device)
            temp_jac = jacobian(jac_encode, img)
            image_dim = 3 * 32 * 32
            temp_jac = temp_jac.reshape(-1, image_dim)
            filename = '{}/model_{}_time_{:d}.pt'.format(image_path, model_name_1, t)
            f = open(filename, 'wb')
            pickle.dump(temp_jac, f)
            f.close()

             # compute the image in the next step
            x_prev = diffusion_step(img, model_1, t, alpha, alpha_next, device)
            img = x_prev
            if i % log_every_t == 0 or i == num_ddim_steps - 1:
                intermediates.append(img)
        
        filename = '{}/model_{}_intermediates.pt'.format(image_path, model_name_1)
        f = open(filename, 'wb')
        pickle.dump(intermediates, f)
        f.close()

        # forward pass for model 2
        img = x0
        log_every_t = 1
        intermediates = [x0]
        time_range = timesteps
        iterator = tqdm(time_range[0:-1], desc='DDIM Forward', total=num_ddim_steps)

        for i, t in enumerate(iterator):
            # compute stepwise jacobian and save the result
            alpha = compute_alpha(betas, time_range[i])
            alpha_next = compute_alpha(betas, time_range[i + 1])
            def jac_encode(img):
                return diffusion_step(img, model_2, t, alpha, alpha_next, device)
            temp_jac = jacobian(jac_encode, img)
            image_dim = 3 * 32 * 32
            temp_jac = temp_jac.reshape(-1, image_dim)
            filename = '{}/model_{}_time_{:d}.pt'.format(image_path, model_name_2, t)
            f = open(filename, 'wb')
            pickle.dump(temp_jac, f)
            f.close()
            
            # compute the image in the next step
            x_prev = diffusion_step(img, model_2, t, alpha, alpha_next, device)
            img = x_prev
            if i % log_every_t == 0 or i == num_ddim_steps - 1:
                intermediates.append(img)
        
        filename = '{}/model_{}_intermediates.pt'.format(image_path, model_name_2)
        f = open(filename, 'wb')
        pickle.dump(intermediates, f)
        f.close()