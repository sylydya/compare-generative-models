import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, TensorDataset

from utils import kl_mvn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)     # Outputs the mean
        self.fc_short_encode = nn.Linear(input_dim, latent_dim)    # shortcut to the latent space

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.fc_short_decode = nn.Linear(latent_dim, input_dim)    # shortcut to the input space

    def encode(self, x):
        """Encodes input by mapping it into the latent space."""
        h1 = F.relu(self.fc1(x))
        mu = self.fc2(h1) + self.fc_short_encode(x)
        return mu


    def decode(self, z):
        """Decodes the latent representation z to reconstruct the input."""
        h3 = F.relu(self.fc3(z))
        # x_recon = torch.sigmoid(self.fc4(h3))
        x_recon = self.fc4(h3) + self.fc_short_decode(z)

        return x_recon

    def forward(self, x):
        """Defines the computation performed at every call."""
        mu = self.encode(x)
        x_recon = self.decode(mu)
        return x_recon, mu

def loss_function(recon_x, x, mu, kl_lambda=0.05):
    """Computes the VAE loss function."""
    # Reconstruction loss (binary cross entropy)
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    MSE = F.mse_loss(recon_x, x, reduction='mean')

    encode_mu = mu.mean(0)
    encode_cov = torch.cov(mu.T)

    # KL divergence between the approximate posterior and the prior
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.05
    
    KLD = kl_lambda * 0.5 * (torch.trace(encode_cov) + torch.dot(encode_mu, encode_mu) - len(encode_mu) - torch.logdet(encode_cov))

    # KLD = 0

    return MSE + KLD



np.random.seed(1)
d = 10
A1_vector = np.random.uniform(0.8, 1.2, d)
A2_vector = A1_vector

A1 = np.diag(A1_vector)
A2 = np.diag(A2_vector)

B1 = np.random.normal(0, 1, d)
B2 = B1


epsilon_list = np.arange(0.0, 0.21, 0.01)

start_index = 0
end_index = 20


for iter_index in range(start_index, end_index):
    seed = iter_index
    epsilon = epsilon_list[seed]

    n = 500000
    d = 10
    np.random.seed(seed)
    A2_vector = A1_vector * (1 + epsilon)

    A1 = np.diag(A1_vector)
    A2 = np.diag(A2_vector)

    B2 = B1 + epsilon


    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, d))
    Y1 = np.random.normal(0, 1, (n, d)) * A1_vector + B1
    Y2 = np.random.normal(0, 1, (n, d)) * A2_vector + B2
    Y = np.random.normal(0, 1, (n, d)) * A1_vector + B1

    Y1 = torch.from_numpy(Y1).float()
    Y2 = torch.from_numpy(Y2).float()
    Y = torch.from_numpy(Y).float()

    d1 = d
    d2 = d

    mu = B1
    sigma = np.power(A1, 2)

    dist = -kl_mvn((mu, sigma), (B1, np.power(A1, 2))) + kl_mvn((mu, sigma), (B2, np.power(A2, 2)))
    print(dist)

    # Hyperparameters
    input_dim = Y1.size(1)   # Dimension of the input data
    hidden_dim = 100        # Size of the hidden layer
    latent_dim = 10         # Dimension of the latent space
    batch_size = 5000        # Batch size for training
    learning_rate = 5e-3    # Learning rate
    epochs = 20             # Number of training epochs

    # Create DataLoader
    epochs = 20             # Number of training epochs
    model1_path = './result/vae/model1.pt'
    
    if os.path.exists(model1_path):
        print('load model 1')
        model1 = VAE(input_dim, hidden_dim, latent_dim)
        model1.load_state_dict(torch.load(model1_path))

    else:
        print('train model 1')
        dataset1 = TensorDataset(Y1)
        dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    
        # Initialize the VAE model
        model1 = VAE(input_dim, hidden_dim, latent_dim)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate, weight_decay=1e-4)
    
        # Training loop
        for epoch in range(epochs):
            model1.train()
            train_loss = 0
            for data_batch in dataloader1:
                data = data_batch[0]
                optimizer1.zero_grad()
                recon_batch, mu = model1(data)
                loss = loss_function(recon_batch, data, mu)
                # loss = loss / batch_size
                loss.backward()
                train_loss += loss.item() * batch_size
                optimizer1.step()
            average_loss = train_loss / len(dataset1)
            print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')
        torch.save(model1.state_dict(), model1_path)

    # Create DataLoader
    model2_path = './result/vae/model2.pt'
    optimizer2_path = './result/vae/optimizer2.pt'
    
    if os.path.exists(model2_path):
        print('load model 2')
        model2 = VAE(input_dim, hidden_dim, latent_dim)
        model2.load_state_dict(torch.load(model2_path))
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    
        # optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)
        # optimizer2.load_state_dict(torch.load(optimizer2_path))
        # with open(optimizer2_path, 'rb') as f:
        #     optimizer2  = pickle.load(f)
        epochs = 20

        print((Y - model1.decode(model1.encode(Y))).pow(2).sum(1).mean())
        print((Y2 - model1.decode(model1.encode(Y2))).pow(2).sum(1).mean())
        print((Y1 - model1.decode(model1.encode(Y1))).pow(2).sum(1).mean())

        print((Y - model2.decode(model2.encode(Y))).pow(2).sum(1).mean())
        print((Y2 - model2.decode(model2.encode(Y2))).pow(2).sum(1).mean())
        print((Y1 - model2.decode(model2.encode(Y1))).pow(2).sum(1).mean())
    else:
        print('print model 2')
        # Initialize the VAE model
        model2 = VAE(input_dim, hidden_dim, latent_dim)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)
        # optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        epochs = 20
    
    # Create DataLoader
    dataset2 = TensorDataset(Y2)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        model2.train()
        train_loss = 0
        for data_batch in dataloader2:
            data = data_batch[0]
            optimizer2.zero_grad()
            recon_batch, mu = model2(data)
            loss = loss_function(recon_batch, data, mu)
            # loss = loss / batch_size
            loss.backward()
            train_loss += loss.item() * batch_size
            optimizer2.step()
        average_loss = train_loss / len(dataset2)
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss:.4f}')
        
    if not os.path.exists(model2_path):
        torch.save(model2.state_dict(), model2_path)

    num_samples = 1000

    def g1_inverse_function(x):
        mu = model1.encode(x)
        return mu

    def g2_inverse_function(x):
        mu = model2.encode(x)
        return mu

    delta_list = []
    std_list = []
    for seed in range(500):
        if seed % 10 == 0:
            print(seed)
        np.random.seed(seed + iter_index * 500)
        Y = np.random.normal(0, 1, (num_samples, d)) * A1_vector + B1
        Y = torch.from_numpy(Y).float()

        g1_inverse = model1.encode(Y).data
        g2_inverse = model2.encode(Y).data
        d1 = d
        d2 = d

        log_jacobian_list = []
        for index in range(num_samples):
            z = torch.clone(Y[index:index+1,].data)
            z.requires_grad = True
            J_g1_inverse = jacobian(g1_inverse_function, Y[index:index+1,].data)
            J_g2_inverse = jacobian(g2_inverse_function, Y[index:index+1,].data)
            log_jacobian_list.append(torch.log(torch.det(J_g1_inverse[0,:,0,:]).abs()).item() - torch.log(torch.det(J_g2_inverse[0,:,0,:]).abs()).item())

        delta_pp = torch.FloatTensor(log_jacobian_list) -0.5 * np.power(g1_inverse, 2).sum(1)[0:num_samples]+ 0.5 * np.power(g2_inverse, 2).sum(1)[0:num_samples] + (d2 - d1) * np.log(2 * np.pi)
        std_pp = delta_pp.std()
        delta_pp = delta_pp.mean()

        delta_list.append(delta_pp)
        std_list.append(std_pp)

    filename = 'result/VAE_coverage_epsilon_{:.3f}.txt'.format(epsilon)
    with open(filename, 'wb') as f:
        pickle.dump([delta_list, std_list, dist], f)