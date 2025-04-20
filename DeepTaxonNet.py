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
import utils
from encoders import *
from decoders import *

class TaxonomicLayer(nn.Module):
    def __init__(self, n_hidden, n_clusters):
        super(TaxonomicLayer, self).__init__()
        self.n_hidden = n_hidden
        self.n_clusters = n_clusters

        self.cluster_weight = nn.Parameter(torch.zeros(n_clusters, 1))

        self.sigmoid = nn.Sigmoid()

    def kl_div(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two Gaussian distributions.
        Args:
            mean1 (Tensor): Mean of the first distribution.
            logvar1 (Tensor): Log variance of the first distribution.
            mean2 (Tensor): Mean of the second distribution.
            logvar2 (Tensor): Log variance of the second distribution.
        Returns:
            Tensor: KL divergence between the two distributions. Shape: (batch_size,)
        """
        return 0.5 * torch.sum(logvar2 - logvar1 + (logvar1.exp() + (mean1 - mean2)**2) / logvar2.exp() - 1, dim=1)

    def forward(self, mean_leaf, logvar_leaf, eps=1e-8):
        """
        Args:
            mean_leaf (Tensor): Leaf means with shape (n_clusters*2, n_hidden).
            var_leaf (Tensor): Leaf variances (diagonal covariances) with shape (n_clusters*2, n_hidden).
        
        Returns:
            mean_parent (Tensor): Parent means with shape (n_clusters, n_hidden).
            var_parent (Tensor): Parent variances with shape (n_clusters, n_hidden).
        """
        # alpha = self.sigmoid(self.cluster_weight) # shape: n_clusters, 1
        # want a more flat distribution. What to do?
        # use a temprature to control the flatness of the distribution
        alpha = self.sigmoid(self.cluster_weight) # shape: n_clusters, 1
        alpha = torch.cat((alpha, 1-alpha), dim=1).unsqueeze(-1) # shape: n_clusters, 2, 1

        mean_leaf_expanded = mean_leaf.view(-1, 2, self.n_hidden) # shape: n_clusters, 2, n_hidden
        logvar_leaf_expanded = logvar_leaf.view(-1, 2, self.n_hidden) # shape: n_clusters, 2, n_hidden

        # pick the two children per cluster
        m_child0, m_child1 = mean_leaf_expanded[:,0], mean_leaf_expanded[:,1]      # each (n_clusters, n_hidden)
        lv_child0, lv_child1 = logvar_leaf_expanded[:,0], logvar_leaf_expanded[:,1]

        # forward and reverse KL
        kl_0_1 = self.kl_div(m_child0, lv_child0, m_child1, lv_child1)  # (n_clusters,)
        kl_1_0 = self.kl_div(m_child1, lv_child1, m_child0, lv_child0)  # (n_clusters,)
        # symmetrized KL and reshape to (n_clusters, 1)
        sym_kl = (kl_0_1 + kl_1_0).unsqueeze(1)  # (n_clusters, 1)

        mean_root = (alpha * mean_leaf_expanded).sum(dim=1) # shape: n_clusters, n_hidden
        var_leaf = torch.exp(logvar_leaf_expanded)

        mean_root_expand = mean_root.unsqueeze(1) # shape: n_clusters, 1, n_hidden
        diff_sq = (mean_leaf_expanded - mean_root_expand)**2 # shape: n_clusters, 2, n_hidden
        var_root = (alpha * (var_leaf + diff_sq)).sum(dim=1) # shape: n_clusters, n_hidden

        logvar_root = torch.log(var_root + eps) # shape : n_clusters, n_hidden

        # compute DKL between leaf and parent
        mean_root_dkl = mean_root.unsqueeze(1).expand(-1, 2, -1).reshape(-1, self.n_hidden) # shape: n_clusters*2, n_hidden
        # print(mean_root_dkl)
        # print(mean_leaf.shape)
        logvar_root_dkl = logvar_root.unsqueeze(1).expand(-1, 2, -1).reshape(-1, self.n_hidden) # shape: n_clusters*2, n_hidden

        dkl = 0.5 * torch.sum(
            logvar_root_dkl - logvar_leaf - 1 + torch.exp(logvar_leaf - logvar_root_dkl) + (mean_leaf - mean_root_dkl)**2 / torch.exp(logvar_root_dkl),
            dim=1
        ) # shape: n_clusters*2, n_hidden
        # print(dkl)

        

        # print(alpha)
        # mean_root = mean_leaf * alpha # shape: n_clusters, 2, n_hidden
        # mean_root = mean_root.reshape(-1, 2, self.n_hidden).sum(dim=1) # shape: n_clusters, n_hidden
        # print(mean_root)
        return mean_root, logvar_root, alpha.reshape(-1), sym_kl # shape: 2 * n_clusters, n_hidden

class DeepTaxonNet(nn.Module):
    def __init__(self, input_dim=3*32*32, 
                 enc_hidden_dim=128,
                 dec_hidden_dim=128, # tuple for ResNet
                 latent_dim=32, 
                 n_layers=5, 
                 encoder_name=None, # string
                 decoder_name=None, # string
                 kl1_weight=1.0,
                 recon_weight=1.0,
                 noise_strength=0.0,
                 ):
        super(DeepTaxonNet, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.n_layers = n_layers
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.kl1_weight = kl1_weight
        self.recon_weight = recon_weight
        self.noise_strength = 0.0

        ## Encoder
        if self.encoder_name == "mlp":
            self.encoder = MLPEncoder(self.input_dim, self.enc_hidden_dim)
        elif self.encoder_name == "resnet18":                 # For images size >= 64x64
            self.encoder = ResNet18Encoder()    
        elif self.encoder_name == "resnet34":                 # For images size >= 64x64
            self.encoder = ResNet34Encoder()
        elif self.encoder_name == "resnet18_light":           # For images size < 64x64
            self.encoder = ResNet18LightEncoder()
        elif self.encoder_name == "resnet20_light":           # For images size < 64x64
            self.encoder = ResNet20LightEncoder()
        elif self.encoder_name == "vgg": 
            self.encoder = VGGEncoder()
        elif self.encoder_name == "mnist":
            self.encoder = Encoder28x28()
        elif self.encoder_name == "omniglot":
            self.encoder = OmniglotEncoder()
        else:
            raise ValueError("Unknown encoder type")
        
        ## Decoder
        if self.decoder_name == "mlp":
            self.decoder_raw = MLPDecoder(self.latent_dim, self.dec_hidden_dim, self.input_dim)
        elif self.decoder_name == "resnet18":                 # For images size >= 64x64
            self.decoder_raw = ResNet18Decoder()
        elif self.decoder_name == "resnet34":                 # For images size >= 64x64
            self.decoder_raw = ResNet34Decoder()
        elif self.decoder_name == "resnet18_light":           # For images size < 64x64
            self.decoder_raw = ResNet18LightDecoder()
        elif self.decoder_name == "resnet20_light":           # For images size < 64x64
            self.decoder_raw = ResNet20LightDecoder()
        elif self.decoder_name == "vgg":                      # For images size < 64x64
            self.decoder_raw = VGGDecoder()
        elif self.decoder_name == "mnist":
            self.decoder_raw = Decoder28x28()
        elif self.decoder_name == "omniglot":
            self.decoder_raw = OmniglotDecoder()
        else:
            raise ValueError("Unknown decoder type")

        ## Projection heads
        self.fc_mu = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(self.enc_hidden_dim),
            # or layernorm? Which one is better?
            # The layernorm is better for the VAE
            nn.LayerNorm(self.enc_hidden_dim),
            nn.Linear(self.enc_hidden_dim, self.latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.enc_hidden_dim),
            nn.Linear(self.enc_hidden_dim, self.latent_dim),
        )

        # self.fc_mu.apply(lambda m: self.layer_init(m, 1e-3))
        # self.fc_logvar.apply(lambda m: self.layer_init(m, 5e-3))

        ## De-Projection heads
        if self.decoder_name == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.dec_hidden_dim),
                nn.ReLU(),
                # nn.Unflatten(1, (self.dec_hidden_dim)),
                self.decoder_raw
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.dec_hidden_dim[0] * self.dec_hidden_dim[1] * self.dec_hidden_dim[2]),
                nn.ReLU(),
                nn.Unflatten(1, (self.dec_hidden_dim[0], self.dec_hidden_dim[1], self.dec_hidden_dim[2])),
                self.decoder_raw
            )

        # init_range = 1e-5  # or compute your desired range based on a fan_in value
        # first_linear = self.decoder[0]  # Access the first element, which is the nn.Linear layer

        # nn.init.uniform_(first_linear.weight, -init_range, init_range)
        # if first_linear.bias is not None:
        #     nn.init.constant_(first_linear.bias, 0.0)

        # GMM Prior Parameters:
        # 1. Cluster prior logits; softmax gives pi (mixing coefficients)
        self.pi_logits = nn.Parameter(torch.zeros(2**(self.n_layers+1)-1))  # Init 0 => softmax gives uniform distribution # shape: (n_clusters,)
        self.pi_logits_baseline = nn.Parameter(torch.zeros(2**self.n_layers))  # Init 0 => softmax gives uniform distribution # shape: (n_clusters,)
        # 2. Cluster means: shape (n_clusters, latent_dim)
        limit = 1 / (2 ** self.n_layers)
        # limit = 0.1
        # limit = 1
        # limit = 5
        # limit = 0
        # limit = np.sqrt(7)
        self.mu_c = nn.Parameter(torch.nn.init.uniform_(torch.empty(2**self.n_layers, self.latent_dim), -limit, limit))  # Init uniform
        # initialize with normal distribution
        # self.mu_c = nn.Parameter(torch.nn.init.normal_(torch.empty(2**self.n_layers, self.latent_dim), 0, 0.6))  # Init uniform

        limit = -3
        self.logvar_c = nn.Parameter(torch.nn.init.uniform_(torch.empty(2**self.n_layers, self.latent_dim), limit, limit+1))  # Init uniform
        # self.mu_c = nn.Parameter(torch.zeros(2**self.n_layers, self.latent_dim))  # Init uniform

        # self.logvar_c = nn.Parameter(torch.zeros(2**self.n_layers, self.latent_dim))  # Init uniform
        self.layers = nn.ModuleList(
            [TaxonomicLayer(self.latent_dim, 2**i) for i in reversed(range(0, self.n_layers))]
        ) 

        self.logvar_x = nn.Parameter(torch.zeros(1)[0])

    def layer_init(self, module, init_range):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -init_range, init_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def pretrain_autoencoder(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        recon_loss = F.mse_loss(x_recon, x)
        return recon_loss

    def encode(self, x):
        h = self.encoder(x)
        # print(f"encoder output range: {h.min()} {h.max()}")
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # print(f"logvar range: {logvar.min()} {logvar.max()}")
        return mu, logvar   
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if not self.training:
            # print("eval")
            eps = torch.zeros_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder(z)
        if self.decoder_name == "mlp":
            x = x.view(-1, 3, 32, 32)
        return x
    

    def forward(self, x, tau=1.0):
        mu, logvar = self.encode(x) # shape: (batch_size, latent_dim)
        # mu = (mu - mu.min()) / mu.std() # normalize mu
        # range for logvar
        # print(f"logvar range: {logvar.min()} {logvar.max()}")
        # print(f"mu range: {mu.min()} {mu.max()}")
        z = self.reparameterize(mu, logvar)
        # z = mu
        x_recon = self.decode(z)
        # print(x_recon)
        # print the range of z
        # print(f"z range: {z.min()} {z.max()}")
        
        # ELBO Loss
        pi, mu_c, logvar_c, H, alphas, dkl_list = self.gmm_params() # shape: (n_clusters, latent_dim)


        # pi = F.softmax(self.pi_logits_baseline, dim=0)
        # mu_c = self.mu_c
        # logvar_c = self.logvar_c

        # pi = F.softmax(self.pi_logits_baseline, dim=0) # shape: (n_clusters,)
        # mu_c = self.mu_c # shape: (n_clusters, latent_dim)
        # logvar_c = self.logvar_c # shape: (n_clusters, latent_dim)

        # root_mean = mu_c[-1, :] # shape: (latent_dim,)
        # root_logvar = logvar_c[-1, :] # shape: (latent_dim,)
        # root kl to N(0, 1)
        # KL_ROOT = -0.5 * torch.sum(1 + root_logvar - root_mean.pow(2) - root_logvar.exp())

        # mu_c = mu_c[:-1, :] # remove the root
        # logvar_c = logvar_c[:-1, :]

        ########################################################
        # 1. reconstruction loss
        ##########################################################
        # check range of x_recon
        # if x_recon.max() > 1 or x_recon.min() < 0:
        #     print("x_recon out of range")
        # try:
            # recon_loss = F.binary_cross_entropy(x_recon, x)
        # except:
        #     print("x_recon out of range")
        #     print(x_recon.max(), x_recon.min())
        #     print(x_recon)

        recon_loss = F.mse_loss(x_recon, x)
        recon_loss = recon_loss * self.input_dim
        recon_loss = recon_loss * self.recon_weight
        # log_sigma = self.softclip(self.logvar_x, -6)
        # recon_loss = self.gaussian_nll(x_recon, log_sigma, x).sum()
        # recon_loss = F.mse_loss(x_recon, x) * self.input_dim / (2 * torch.exp(log_sigma)) + log_sigma * self.input_dim / 2


        # why multiply by input_dim?
        # because recon_loss is averaged over the batch and the input dimension

        #####################################################
        # 1.5 calculate q(c|x) as approximated by p(c|z)
        #####################################################
        # p(c|z) is p(c)p(z|c)/sum(p(z|c)p(c))
        z_l = self.reparameterize(mu, logvar) # sample a new z
        log_pdf = self.gaussian_pdf(z_l, mu_c, logvar_c) # shape: (batch_size, n_clusters). This is p(z|c)
        # tau = 0.8
        if not self.training:
            # print("eval")
            alpha = 0.0
        else:
            alpha = self.noise_strength
        # print("alpha", alpha)
        logpcpzc = (log_pdf + torch.log(pi.unsqueeze(0))) # log p(c|z) = log p(c) + log p(z|c)
        pcx, logpcx = untils.GumbelSoftmax(logpcpzc, # Gumbel softmax for exploration
                                            tau=tau, 
                                            alpha=alpha,
                                            hard=False, # MUST be False
                                            dim=-1)
        # pcpzc = (log_pdf + torch.log(pi.unsqueeze(0))) / tau
        # logpcx = pcpzc - torch.logsumexp(pcpzc, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        # pcx = torch.exp(logpcx) # shape: (batch_size, n_clusters)

        # print(pcx)
        # logpzc = log_pdf - torch.logsumexp(log_pdf, dim=1, keepdim=True)
        # print(torch.exp(logpzc))
        # exit(0)
        # logcollocation = logpcx + log_pdf
        # logcollocation = logcollocation - torch.logsumexp(logcollocation, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        # pcx = torch.exp(logcollocation) # shape: (batch_size, n_clusters)
        
        # print(pcx[0])
        # find p(z|c)*p(c|x)
        # print((torch.exp(log_pdf - torch.logsumexp(log_pdf, dim=1, keepdim=True)) * pcx).argmax(dim=1)[200:300])
        # print(pcx.argmax(dim=1)[200:300])
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

        #############################################################
        # 2: KL divergence between q(z|x) and p(z|c)
        #############################################################
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
        kl1 = kl1 * self.kl1_weight

        ###############################################################
        # 3: KL divergence between p(c|x) and p(c)
        ###############################################################
        kl2 = torch.sum(
            pcx * (torch.log(pi.unsqueeze(0)) - logpcx),
            dim=1
        )
        kl2 = torch.mean(kl2)
        kl2 = kl2 * self.kl1_weight

        # print("recon_loss", recon_loss.item(),
        #       "kl1", -kl1.item(),
        #       "kl2", -kl2.item())

        # total loss


        loss = recon_loss - kl1 - kl2

        # print("alphas", alphas)

        # encourgae alphas to be uniform
        alphas = alphas / alphas.sum(dim=0, keepdim=True)
        # compare to uniform distribution
        U = torch.ones_like(alphas) / alphas.shape[0]

        # compute the kl between the alpha distribution and the uniform distribution
        kl_alpha = torch.sum(alphas * (torch.log(alphas + 1e-10) - torch.log(U + 1e-10)), dim=0)
        # print("U", U)
        # print("kl_alpha", kl_alpha.item())
        # exit(0)

        loss = loss + kl_alpha



        # # penalize the high entropy of the leave node
        H_leaf = H[0]
        H_root = H[-1]
        # loss = loss# + 0.01 * torch.mean(H_leaf) - 0.01 * torch.mean(H_root)
        # H_leaf = H_leaf.sum(dim=0)
        # print("H_leaf", H_leaf)
        # clamp the H_leaf with maximum of 0.1
        # self.logvar_c = torch.clamp(self.logvar_c, max=0.1)
        # H_leaf = torch.clamp(H_leaf, max=0.1)
        # # print("H_leaf", H_leaf)
        # T = H_leaf.mean()
        # penalty = torch.relu(H_leaf - T)
        # entropy_reg_loss = 1 * torch.mean(penalty)
        # loss = loss + entropy_reg_loss
        # print("entropy_reg_loss", entropy_reg_loss.item())

        # compute information gain
        # logpdf to H
        # H_node = - torch.exp(log_pdf) * log_pdf
        # # print("H_node", H_node.shape)
        # H_root = H_node[:, -1:]
        # # print("H_root", H_root.shape)
        # H_leaf = H_node[:, :(2**self.n_layers)]
        # H_parents = H_node[:, (2**self.n_layers):]
        # IG = H_root.sum(dim=1) + H_parents.sum(dim=1) - H_leaf.sum(dim=1)
        # # print("IG", IG, IG.shape)
        # IG = torch.mean(IG)
        # print("IG", IG.item())
        # loss = loss - IG

        dkl_list_flatten = torch.cat(dkl_list, dim=0) / 2# shape: (2 ** (n_layers + 1) - 2, n_clusters)
        # print(dkl_list_flatten)
        margin = 1
        dkl_penalty = torch.relu(margin - dkl_list_flatten)
        # print("dkl_penalty", dkl_penalty.sum().item())
        loss = loss + dkl_penalty.sum()

        return loss, recon_loss, kl1, kl2, H, pcx, pi, dkl_list
    
    def softclip(self, tensor, min):
        """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
        result_tensor = min + F.softplus(tensor - min)

        return result_tensor
    def gaussian_nll(self, mu, log_sigma, x):
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)
    
    def gmm_params(self):   # Return the cluster prior probabilities (pi), means, and log variances.
        layer_mu = []
        layer_logvar = []
        pi_list = []
        dkl_list = []

        layer_mu.append(self.mu_c)
        layer_logvar.append(self.logvar_c)

        parent_root = self.mu_c
        parent_logvar = self.logvar_c

        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar, alpha, dkl = layer(parent_root, parent_logvar)
            layer_mu.append(parent_root)
            layer_logvar.append(parent_logvar)
            pi_list.append(alpha)
            dkl_list.append(dkl)

        # pi_list.append(torch.tensor([1.0], device=parent_root.device))
        # normalize pi_list by its parent
        # pi_list = pi_list[::-1] # reverse the list
        # for pl in pi_list:
        #     print(pl.shape)
        # print("pi_list", pi_list)
        # chain the probabilities from root to leaves
        # for i in range(len(pi_list)-1):
        #     pi_list[i+1] = pi_list[i].unsqueeze(1).expand(-1, 2).reshape(-1) * pi_list[i+1]
        # reverse the list again
        # pi_list = pi_list[::-1][:-1] # remove the last element
        # concat
        alphas = torch.cat(pi_list, dim=0) # shape: (2 ** (n_layers + 1) - 1, n_clusters)
        # print("alphas", alphas)
        # pi = pi / pi.sum()
        # print("pi_list", pi)
        # exit()

        # get entropy at each layer
        H = []
        for logvar_layer in layer_logvar:
            H.append(0.5 * torch.sum(logvar_layer, dim=1))

        # Concatenate the means and log variances from all layers
        mu_c = torch.cat(layer_mu, dim=0)  # shape: (2 ** (n_layers + 1) - 1, latent_dim)
        logvar_c = torch.cat(layer_logvar, dim=0)
        # pi = F.softmax(self.pi_logits, dim=0)  # (n_clusters,)

        # dkl_list = torch.cat(dkl_list, dim=0) # shape: (2 ** (n_layers + 1) - 2, n_clusters)
        # dkl_list = dkl_list / dkl_list.sum()
        # dkl_list = F.softmax(dkl_list, dim=0) # (n_clusters,)

        # pi = dkl_list
        # check if pi has any zeros
        # if torch.any(pi == 0):
            # print("pi has zeros")
        # mu_c = mu_c[:-1, :] # remove the root
        # logvar_c = logvar_c[:-1, :]

        # print(mu_c.shape, logvar_c.shape, pi.shape)
        # dkl_list = F.softmax(dkl_list, dim=0) # (n_clusters,)

        # print("dkl_list", dkl_list)
        # mu_c = self.mu_c
        # logvar_c = self.logvar_c
        # pi = F.softmax(self.pi_logits, dim=0) # shuould give uniform distribution
        # pi = torch.tensor([1/2**(self.n_layers)] * (2**self.n_layers), device=pi.device)
        N_NODES = 2**(self.n_layers+1)-1
        pi = torch.tensor([1/N_NODES] * N_NODES, device=mu_c.device)
        # print("pi", pi.shape, pi)
        # exit()
        return pi, mu_c, logvar_c, H, alphas, dkl_list
    
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
    
    def vae_forward(self, x): # for testing
        # VAE loss
        mu, logvar = self.encode(x) # shape: (batch_size, latent_dim)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        loss_recon = F.mse_loss(x_recon, x)
        loss_recon = loss_recon * self.input_dim
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss_recon + kld
        return loss, loss_recon, kld, 0

    def elbo_kl(self, z_l, pi, mu_c, logvar_c): # deprecated. The Multi-Facet VaDE style loss
        # z = self.reparameterize(mu, logvar)

        # ELBO Loss
        # x_recon = self.decode(z)
        # recon_loss = F.mse_loss(x_recon, x) * self.input_dim

        # calculate p(c|x) as approximated by p(c|z)
        log_pdf = self.gaussian_pdf(z_l, mu_c, logvar_c) # shape: (batch_size, n_clusters) this is p(z|cj)
        log_pdf = log_pdf - torch.logsumexp(log_pdf, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        pcpzc = log_pdf + torch.log(pi.unsqueeze(0))
        logpcx = pcpzc - torch.logsumexp(pcpzc, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        pcx = torch.exp(logpcx) # shape: (batch_size, n_clusters)

        # second term: KL divergence between q(z|x) and p(z|c)
        # qzxlogpzc = torch.sum( # summing over the latent dimensions
        #             (logvar_c.unsqueeze(0) + torch.exp(logvar.unsqueeze(1) - logvar_c.unsqueeze(0)) +
        #            (mu.unsqueeze(1) - mu_c.unsqueeze(0))**2 / torch.exp(logvar_c.unsqueeze(0))),
        #     dim=2
        # )   # shape: (batch_size, n_clusters)
        # E_logpzc = -0.5 * torch.sum(qzxlogpzc * pcx, dim=1) # shape : (batch_size,)
        # E_logqzx = -torch.sum( # summing over the latent dimensions
        #     0.5 * (1 + logvar),
        #     dim=1
        # )   # shape: (batch_size,)
        # kl1 = torch.mean(E_logpzc - E_logqzx)

        # third term: KL divergence between p(c|x) and p(c)
        # kl2 = torch.sum(
        #     pcx * (torch.log(pi.unsqueeze(0)) - logpcx),
        #     dim=1
        # )
        # kl2 = torch.mean(kl2)

        # loss = recon_loss - kl1 - kl2
        return pcx

    def _forward(self, x): # deprecated. The Multi-Facet VaDE style loss

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


        mu_c_list, logvar_c_list, pi_list, pi_list_raw = [], [], [], []
        kl1_list, kl2_list = [], []
        pcx_j = []

        parent_root = self.mu_c
        parent_logvar = self.logvar_c

        mu_c_list.append(parent_root)
        logvar_c_list.append(parent_logvar)

        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar, pi, dkl = layer(parent_root, parent_logvar)
            mu_c_list.append(parent_root)
            logvar_c_list.append(parent_logvar)
            pi_list.append(pi)
            pi_list_raw.append(pi)

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
        # print("pi_list", pi_list)

        z_l = self.reparameterize(mu, logvar) # sample a new z
        for i in range(self.n_layers):
            pcx = self.elbo_kl(z_l, pi_list[i], mu_c_list[i], logvar_c_list[i])
            pcx_j.append(pcx)

            # kl1_list.append(kl1)
            # kl2_list.append(kl2)

        # print("pcx_j", pcx_j)
        # print("kl2 list", kl2_list)

        pcx_j = torch.cat(pcx_j, dim=1)
        # normalize pcx_j
        pcx_j = pcx_j / torch.sum(pcx_j, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        # softmax 
        # pcx_j = F.softmax(pcx_j, dim=1) # shape: (batch_size, n_clusters)
        # pcx_j = F.softmax(pcx_j, dim=1) # shape: (batch_size, n_clusters)
        # print("pcx_j", pcx_j[0])
        mu_c = torch.cat(mu_c_list, dim=0)[:-1]  # shape: (2 ** (n_layers + 1) - 1, latent_dim)
        logvar_c = torch.cat(logvar_c_list, dim=0)[:-1] # shape: (2 ** (n_layers + 1) - 1, latent_dim)
        qzxlogpzc = torch.sum( # summing over the latent dimensions
                    (logvar_c.unsqueeze(0) + torch.exp(logvar.unsqueeze(1) - logvar_c.unsqueeze(0)) +
                   (mu.unsqueeze(1) - mu_c.unsqueeze(0))**2 / torch.exp(logvar_c.unsqueeze(0))),
            dim=2
        )   # shape: (batch_size, n_clusters)
        E_logpzc = -0.5 * torch.sum(qzxlogpzc * pcx_j, dim=1) # shape : (batch_size,)
        E_logqzx = -torch.sum( # summing over the latent dimensions
            0.5 * (1 + logvar),
            dim=1
        )   # shape: (batch_size,)
        kl1 = torch.mean(E_logpzc - E_logqzx)

        # pi_list_raw = torch.cat(pi_list_raw, dim=0)
        # normalize pi_list_raw
        # pi_list_raw = pi_list_raw / torch.sum(pi_list_raw, dim=0, keepdim=True) # shape: (n_clusters,)
        # print("pi_list_raw", pi_list_raw)
        # print(pcx_j[0])
        pi_list_raw = F.softmax(self.pi_logits, dim=0)[:-1]
        kl2 = torch.sum(
            pcx_j * (torch.log(pi_list_raw.unsqueeze(0)) - torch.log(pcx_j + 1e-10)),
            dim=1
        )
        kl2 = torch.mean(kl2)
        # print("kl2", kl2.item())

        # print("kl1_list", kl1_list)
        # print("kl2_list", kl2_list)


        # print(len(kl1_list), len(kl2_list))

        # kl1_sum = torch.mean(torch.stack(kl1_list))
        # kl2 = torch.mean(torch.stack(kl2_list))

        loss = recon_loss - kl1 - kl2

        return loss, recon_loss, pcx_j, kl2

    