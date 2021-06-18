import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VQ_VAE.VQ_VAE import vq_vae
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torchvision.utils as vutils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

#data_variance = np.var(training_data.data / 255.0)

batch_size = 256
num_training_updates = 55000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3


training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)


model = vq_vae(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_res_recon_error = []
train_res_perplexity = []
save_dir = os.getcwd()

for i in range(num_training_updates):
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()

    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i + 1) % 100 == 0:
        print('%d iterations' % (i + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()

    if (i + 1) % 1000 == 0:
        with torch.no_grad():
            (valid_originals, _) = next(iter(validation_loader))
            valid_originals = valid_originals.to(device)

            _, recon_data, _ = model(valid_originals)

            vutils.save_image(recon_data[0:16].cpu(),
                              "%s/%s/%s_%d_%d.png" % (save_dir, "Results", 28, 28, i + 1))

