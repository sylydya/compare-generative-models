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
from pytorch_diffusion import Diffusion
from torch.autograd.functional import jacobian


parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--S', default=50, type=int, help="number of denoising step")
parser.add_argument('--skip', default='linear', type = str, help = 'method to select the subsequence of time steps')

args = parser.parse_args()


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


# CIFAR10 Data
data_path = "./data/"
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# path to save jacobian reesults
data_name = 'cifar10'
save_result_path = "./result/{}".format(data_name)

if not os.path.isdir(save_result_path):
    try:
        os.makedirs(save_result_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(PATH):
            pass
        else:
            raise
# DDIM Setting
num_ddim_steps = args.S
num_diffusion_timesteps = 1000
if args.skip == 'linear':
    timesteps = torch.ceil(torch.linspace(0, num_diffusion_timesteps - 1, num_ddim_steps + 1)).long().to(device)
else:
    timesteps = torch.linspace(0, np.sqrt(num_diffusion_timesteps - 1) , num_ddim_steps + 1).pow(2).round().long().to(device)

beta_start = 0.0001
beta_end = 0.02
betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps
        )
betas = torch.from_numpy(betas).float().to(device)


# load pretrained model, ema_cifar10 and cifar10
model_name = "ema_{}".format(data_name)
diffusion_model = Diffusion.from_pretrained(model_name)
model = diffusion_model.model
model.to(device)


for image_index, (x, y) in enumerate(test_loader):
    img = 2 * x - 1
    img = img.to(device)
    x0 = img
    image_path = '{}/S_{}/image_{:d}'.format(save_result_path, num_ddim_steps, image_index)
    if not os.path.isdir(image_path):
        try:
            os.makedirs(image_path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise
    if args.skip == 'linear':
        filename = '{}/model_{}_intermediates.pt'.format(image_path, model_name)
        jac_filename = '{}/model_{}_jac_logdet.pt'.format(image_path, model_name)
    else:
        filename = '{}/model_{}_{}_intermediates.pt'.format(image_path, model_name, args.skip)
        jac_filename = '{}/model_{}_{}_jac_logdet.pt'.format(image_path, model_name, args.skip)
        
        
    if not os.path.exists(filename):
        print(image_index, model_name, y)
        img = x0
        log_every_t = 1
        intermediates = [x0]
        time_range = timesteps
        iterator = tqdm(time_range[0:-1], desc='DDIM Forward', total=num_ddim_steps)
        product_jac = None
        for i, t in enumerate(iterator):
            # compute stepwise jacobian
            alpha = compute_alpha(betas, time_range[i])
            alpha_next = compute_alpha(betas, time_range[i + 1])
            def jac_encode(img):
                return diffusion_step(img, model, t, alpha, alpha_next, device)
            temp_jac = jacobian(jac_encode, img)
            image_dim = 3 * 32 * 32
            temp_jac = temp_jac.reshape(-1, image_dim)
            if product_jac is None:
                product_jac = temp_jac
            else:
                product_jac = product_jac.matmul(temp_jac)

            # compute the image in the next step
            x_prev = diffusion_step(img, model, t, alpha, alpha_next, device)
            img = x_prev
            if i % log_every_t == 0 or i == num_ddim_steps - 1:
                intermediates.append(img)

        if torch.det(product_jac) < 0:
            temp = torch.clone(product_jac)
            temp[0,:] = -1 * temp[0,:]
            jac_logdet_model1 = torch.logdet(temp)
        else:
            jac_logdet_model1 = torch.logdet(product_jac)

        f = open(jac_filename, 'wb')
        pickle.dump(jac_logdet_model1, f)
        f.close()
        
        f = open(filename, 'wb')
        pickle.dump(intermediates, f)
        f.close()
        

