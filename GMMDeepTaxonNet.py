import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import PIL 
from tqdm import tqdm
import matplotlib.pyplot as plt
import untils
from encoder_decoder.resnet_encoder import Encoder, BasicBlockEnc
from encoder_decoder.restnet_decoder import Decoder, BasicBlockDec
from encoder_decoder.resnet_using_light_basic_block_encoder import LightEncoder, LightBasicBlockEnc
from encoder_decoder.resnet_using_light_basic_block_decoder import LightDecoder, LightBasicBlockDec


class CobwebNNTreeLayer(nn.Module):
    def __init__(self, n_hidden, n_clusters):
        super(CobwebNNTreeLayer, self).__init__()
        self.n_hidden = n_hidden
        self.n_clusters = n_clusters

        self.cluster_weight = nn.Parameter(torch.zeros(n_clusters, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, mean_leaf, logvar_leaf, eps=1e-8):
        """
        Args:
            mean_leaf (Tensor): Leaf means with shape (n_clusters*2, n_hidden).
            var_leaf (Tensor): Leaf variances (diagonal covariances) with shape (n_clusters*2, n_hidden).
        
        Returns:
            mean_parent (Tensor): Parent means with shape (n_clusters, n_hidden).
            var_parent (Tensor): Parent variances with shape (n_clusters, n_hidden).
        """
        alpha = self.sigmoid(self.cluster_weight) # shape: n_clusters, 1
        alpha = torch.cat((alpha, 1-alpha), dim=1).unsqueeze(-1) # shape: n_clusters, 2, 1

        mean_leaf = mean_leaf.view(-1, 2, self.n_hidden) # shape: n_clusters, 2, n_hidden
        logvar_leaf = logvar_leaf.view(-1, 2, self.n_hidden) # shape: n_clusters, 2, n_hidden

        mean_root = (alpha * mean_leaf).sum(dim=1) # shape: n_clusters, n_hidden
        var_leaf = torch.exp(logvar_leaf)

        mean_root_expand = mean_root.unsqueeze(1) # shape: n_clusters, 1, n_hidden
        diff_sq = (mean_leaf - mean_root_expand)**2 # shape: n_clusters, 2, n_hidden
        var_root = (alpha * (var_leaf + diff_sq)).sum(dim=1) # shape: n_clusters, n_hidden

        logvar_root = torch.log(var_root + eps)
        # print(alpha)
        # mean_root = mean_leaf * alpha # shape: n_clusters, 2, n_hidden
        # mean_root = mean_root.reshape(-1, 2, self.n_hidden).sum(dim=1) # shape: n_clusters, n_hidden
        # print(mean_root)
        return mean_root, logvar_root, alpha.reshape(-1) # shape: 2 * n_clusters, n_hidden
    

class GMMDeepTaxonNet(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, n_layers=5, encoder_shared=False):
        super(GMMDeepTaxonNet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.encoder_shared = encoder_shared

        # self.encoder = LightEncoder(LightBasicBlockEnc, [2, 2, 2]) 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        if encoder_shared:
            self.fc_mu = nn.Sequential(
                nn.Linear(128, latent_dim),
            )
            self.fc_logvar = nn.Sequential(
                nn.Linear(128, latent_dim),
            )

        else:
            self.fc_mu_list = nn.ModuleList()
            for _ in range(self.n_layers+1):
                self.fc_mu_list.append(
                    nn.Linear(128, latent_dim)
                )
            self.fc_logvar_list = nn.ModuleList()
            for _ in range(self.n_layers+1):
                self.fc_logvar_list.append(
                    nn.Linear(128, latent_dim)
                )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.ReLU(),
            # nn.Sigmoid(),
        )
        # GMM Prior Parameters:
        # 1. Cluster prior logits; softmax gives pi (mixing coefficients)
        # self.pi_logits = nn.Parameter(torch.zeros(n_clusters))
        # self.pi_logits = nn.Parameter(torch.zeros(2**(self.n_layers+1)-1))  # Init 0 => softmax gives uniform distribution # shape: (n_clusters,)
        # 2. Cluster means: shape (n_clusters, latent_dim)
        limit = 1 / (2 ** self.n_layers)
        self.mu_c = nn.Parameter(torch.nn.init.uniform_(torch.empty(2**self.n_layers, self.latent_dim), -limit, limit))  # Init uniform
        self.logvar_c = nn.Parameter(torch.nn.init.uniform_(torch.empty(2**self.n_layers, self.latent_dim), -limit, limit))  # Init uniform
        self.layers = nn.ModuleList(
            [CobwebNNTreeLayer(self.latent_dim, 2**i) for i in reversed(range(0, self.n_layers))]
        ) 
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar   
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def vae_forward(self, x):
        # VAE loss
        mu, logvar = self.encode(x) # shape: (batch_size, latent_dim)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        loss_recon = F.mse_loss(x_recon, x)
        loss_recon = loss_recon * self.input_dim
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss_recon + kld
        return loss, 0, 0, 0

    def elbo_kl(self, mu, logvar, pi, mu_c, logvar_c):
        # z = self.reparameterize(mu, logvar)

        # ELBO Loss
        # x_recon = self.decode(z)
        # recon_loss = F.mse_loss(x_recon, x) * self.input_dim

        # calculate p(c|x) as approximated by p(c|z)
        z_l = self.reparameterize(mu, logvar) # sample a new z
        log_pdf = self.gaussian_pdf(z_l, mu_c, logvar_c) # shape: (batch_size, n_clusters)
        pcpzc = log_pdf + torch.log(pi.unsqueeze(0))
        logpcx = pcpzc - torch.logsumexp(pcpzc, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        pcx = torch.exp(logpcx) # shape: (batch_size, n_clusters)

        # second term: KL divergence between q(z|x) and p(z|c)
        qzxlogpzc = torch.sum( # summing over the latent dimensions
                    (logvar_c.unsqueeze(0) + torch.exp(logvar.unsqueeze(1) - logvar_c.unsqueeze(0)) +
                   (mu.unsqueeze(1) - mu_c.unsqueeze(0))**2 / torch.exp(logvar_c.unsqueeze(0))),
            dim=2
        )   # shape: (batch_size, n_clusters)
        E_logpzc = -0.5 * torch.sum(qzxlogpzc * pcx, dim=1) # shape : (batch_size,)
        E_logqzx = -torch.sum( # summing over the latent dimensions
            0.5 * (1 + logvar),
            dim=1
        )   # shape: (batch_size,)
        kl1 = torch.mean(E_logpzc - E_logqzx)

        # third term: KL divergence between p(c|x) and p(c)
        kl2 = torch.sum(
            pcx * (torch.log(pi.unsqueeze(0)) - logpcx),
            dim=1
        )
        kl2 = torch.mean(kl2)

        # loss = recon_loss - kl1 - kl2
        return kl1, kl2

    def _forward(self, x):

        h = self.encoder(x)

        if self.encoder_shared:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            x_hat = self.decode(z)
            recon_loss = F.mse_loss(x_hat, x)
            recon_loss = recon_loss * self.input_dim

        else:
            mu_list, logvar_list = [], []

            for i in range(self.n_layers+1):
                mu_list.append(self.fc_mu_list[i](h))
                logvar_list.append(self.fc_logvar_list[i](h))

            z_list = []
            for i in range(self.n_layers+1):
                z_list.append(self.reparameterize(mu_list[i], logvar_list[i]))

            x_hat_list = []
            for i in range(self.n_layers+1):
                x_hat_list.append(self.decode(z_list[i]))

            recon_loss_mean = torch.mean(
                torch.stack(
                    [F.mse_loss(x_hat_list[i], x) for i in range(self.n_layers+1)]
                )
            )
            recon_loss = recon_loss_mean * self.input_dim


        mu_c_list, logvar_c_list, pi_list = [], [], []
        kl1_list, kl2_list = [], []

        parent_root = self.mu_c
        parent_logvar = self.logvar_c

        mu_c_list.append(parent_root)
        logvar_c_list.append(parent_logvar)

        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar, pi = layer(parent_root, parent_logvar)
            mu_c_list.append(parent_root)
            logvar_c_list.append(parent_logvar)
            pi_list.append(pi)

        pi_list.append(torch.tensor([1.0], device=h.device)) # should has the length of n_layers+1
        # normalize pi_list by its parent
        pi_list = pi_list[::-1] # reverse the list
        # for pl in pi_list:
        #     print(pl.shape)
        # print("pi_list", pi_list)
        for i in range(len(pi_list)-1):
            pi_list[i+1] = pi_list[i].unsqueeze(1).expand(-1, 2).reshape(-1) * pi_list[i+1]
        # reverse the list again
        pi_list = pi_list[::-1]

        for i in range(self.n_layers+1):
            if self.encoder_shared:
                kl1, kl2 = self.elbo_kl(mu, logvar, pi_list[i], mu_c_list[i], logvar_c_list[i])
            else:
                kl1, kl2 = self.elbo_kl(mu_list[i], logvar_list[i], pi_list[i], mu_c_list[i], logvar_c_list[i])
            kl1_list.append(kl1)
            kl2_list.append(kl2)


        # print(len(kl1_list), len(kl2_list))

        kl1_sum = torch.sum(torch.stack(kl1_list))
        kl2_sum = torch.sum(torch.stack(kl2_list))

        loss = recon_loss - kl1_sum - kl2_sum

        return loss, recon_loss, kl1_sum, kl2_sum

    
    def forward(self, x):
        mu, logvar = self.encode(x) # shape: (batch_size, latent_dim)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        # print(x_recon)
        
        # ELBO Loss
        pi, mu_c, logvar_c = self.gmm_params() # shape: (n_clusters, latent_dim)

        # first term: reconstruction loss
        # check range of x_recon
        # if x_recon.max() > 1 or x_recon.min() < 0:
        #     print("x_recon out of range")
        try:
            # recon_loss = F.binary_cross_entropy(x_recon, x)
            recon_loss = F.mse_loss(x_recon, x)
        except:
            print("x_recon out of range")
            print(x_recon.max(), x_recon.min())
            print(x_recon)
        recon_loss = recon_loss * self.input_dim
        # why multiply by input_dim?
        # because recon_loss is averaged over the batch and the input dimension

        # calculate p(c|x) as approximated by p(c|z)
        # p(c|z) is p(c)p(z|c)/sum(p(z|c)p(c))
        z_l = self.reparameterize(mu, logvar) # sample a new z
        log_pdf = self.gaussian_pdf(z_l, mu_c, logvar_c) # shape: (batch_size, n_clusters)

        # print(log_pdf)
        # print(torch.log(pi.unsqueeze(0)))
        pcpzc = log_pdf + torch.log(pi.unsqueeze(0))
        # pcx = F.softmax(pcpzc/0.0001, dim=1) # shape: (batch_size, n_clusters)
        logpcx = pcpzc - torch.logsumexp(pcpzc, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        pcx = torch.exp(logpcx) # shape: (batch_size, n_clusters)
        
        print(pcx.argmax(dim=1)[200:300])
        # print(pi.shape, pi, pi[:2])
        # normalize p(c|z)
        # pcx = pcpzc / pcpzc.sum(dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        # check if any zeros in pcx
        # if torch.any(pcx == 0):
        #     print("pcx has zeros")
        #     print(pcx)
        #     pcx = pcx + 1e-10
        # any nans in pcx
        # if torch.any(torch.isnan(pcx)):
        #     print("pcx has nans")

        # second term: KL divergence between q(z|x) and p(z|c)
        # q(z|x) and log p(z|c)
        qzxlogpzc = torch.sum( # summing over the latent dimensions
                    (logvar_c.unsqueeze(0) + torch.exp(logvar.unsqueeze(1) - logvar_c.unsqueeze(0)) +
                   (mu.unsqueeze(1) - mu_c.unsqueeze(0))**2 / torch.exp(logvar_c.unsqueeze(0))),
            dim=2
        ) # shape: (batch_size, n_clusters)
        # print(qzxlogpzc)
        # print(pcx)
        E_logpzc = -0.5 * torch.sum(qzxlogpzc * pcx, dim=1) # shape : (batch_size,) 
        # print(E_logpzc)
        E_logqzx = -torch.sum( # summing over the latent dimensions
            0.5 * (1 + logvar),
            dim=1
        ) # shape: (batch_size,)
        kl1 = torch.mean(E_logpzc - E_logqzx)

        # third term: KL divergence between p(c|x) and p(c)
        kl2 = torch.sum(
            pcx * (torch.log(pi.unsqueeze(0)) - logpcx),
            dim=1
        )
        kl2 = torch.mean(kl2)

        # print("recon_loss", recon_loss.item(),
        #       "kl1", kl1.item(),
        #       "kl2", kl2.item())

        # total loss
        loss = recon_loss - kl1 - kl2

        return loss, recon_loss, kl1, kl2
    
    def gmm_params(self):
        # Return the cluster prior probabilities (pi), means, and log variances.
        layer_mu = []
        layer_logvar = []
        pi_list = []

        layer_mu.append(self.mu_c)
        layer_logvar.append(self.logvar_c)

        parent_root = self.mu_c
        parent_logvar = self.logvar_c

        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar, pi = layer(parent_root, parent_logvar)
            layer_mu.append(parent_root)
            layer_logvar.append(parent_logvar)
            pi_list.append(pi)
        pi_list.append(torch.tensor([1.0], device=parent_root.device))
        # normalize pi_list by its parent
        pi_list = pi_list[::-1] # reverse the list
        # for pl in pi_list:
        #     print(pl.shape)
        # print("pi_list", pi_list)
        for i in range(len(pi_list)-1):
            pi_list[i+1] = pi_list[i].unsqueeze(1).expand(-1, 2).reshape(-1) * pi_list[i+1]
        # reverse the list again
        pi_list = pi_list[::-1]
        # concat
        pi = torch.cat(pi_list, dim=0) # shape: (2 ** (n_layers + 1) - 1, n_clusters)
        pi = pi / pi.sum()
        # Concatenate the means and log variances from all layers
        mu_c = torch.cat(layer_mu, dim=0)  # shape: (2 ** (n_layers + 1) - 1, latent_dim)
        logvar_c = torch.cat(layer_logvar, dim=0)
        # pi = F.softmax(self.pi_logits, dim=0)  # (n_clusters,)
        return pi, mu_c, logvar_c
    
    # def forward(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.reparameterize(mu, logvar)
    #     x_recon = self.decode(z)
    #     return x_recon, mu, logvar, z
    
    def gaussian_pdf(self, x, mu, logvar):
        """
        Compute the Gaussian PDF N(x|mu, var) for each dimension of the latent space.
        Args:
            x (Tensor): Input tensor of shape (batch_size, latent_dim).
            mu (Tensor): Mean tensor of shape (n_clusters, latent_dim).
            logvar (Tensor): Log variance tensor of shape (n_clusters, latent_dim).
        Returns:
            Tensor: Gaussian PDF N(x|mu, var) for each dimension of the latent space, shape (batch_size, n_clusters).
                    Meaning the logprob of each cluster for each sample.
        """
        var = torch.exp(logvar) # shape (n_clusters, latent_dim)
        # check if var is zero
            # var = var + 1e-10
        logpdf = -0.5 * (np.log(2*np.pi) + logvar.unsqueeze(0) + (x.unsqueeze(1) - mu.unsqueeze(0))**2 / var.unsqueeze(0))
        return logpdf.sum(-1)
    