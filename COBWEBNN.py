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

## Maximize predictive prower for each node
class CobwebNNTreeLayer(nn.Module):
    def __init__(self, n_hidden, n_clusters):
        super(CobwebNNTreeLayer, self).__init__()
        self.n_hidden = n_hidden
        self.n_clusters = n_clusters

        self.cluster_weight = nn.Parameter(torch.randn(n_clusters, 1))

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

        # mean_root = mean_leaf * alpha # shape: n_clusters, 2, n_hidden
        # mean_root = mean_root.reshape(-1, 2, self.n_hidden).sum(dim=1) # shape: n_clusters, n_hidden

        return mean_root, logvar_root
    
class CobwebNN(nn.Module):
    def __init__(self, n_hidden, n_layers, imprint_dim):
        super(CobwebNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.imprint_dim = imprint_dim

        self.leaves = nn.Parameter(torch.randn(2**n_layers, n_hidden))
        self.leaves_logvar = nn.Parameter(torch.randn(2**n_layers, n_hidden))

        self.layers = nn.ModuleList(
            [CobwebNNTreeLayer(n_hidden, 2**i) for i in reversed(range(0, n_layers))]
        )

    def sample(self, mean, logvar):
        '''
        Reparameterization trick for sampling from Gaussian
        N(μ, σ) = μ + σ * ε
        where ε ~ N(0, 1)
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def p_x_node(self, x, mean, logvar):
        '''
        Calculate the log-likelihood of x given the node
        x: [batch_size, n_hidden]
        mean: [n_clusters, n_hidden]
        '''
        x = x.unsqueeze(1)
        mean = mean.unsqueeze(0)
        return -0.5 * (torch.log(2 * torch.tensor(np.pi)) + logvar + (x - mean)**2 / torch.exp(logvar)).sum(dim=-1)
    
    def forward(self, x, y=None, hard=None):
        x = x.view(-1, self.n_hidden)
        B = x.size(0)
        
        parent_root = self.leaves
        parent_logvar = self.leaves_logvar
        means = []
        logvars = []
        sampled = []
        p_x_nodes = []
        layer_logits = []

        # means.append(parent_root)
        # logvars.append(parent_logvar)

        # layer_logits.append(torch.matmul(x, parent_root.T))

        x_sampled = self.sample(parent_root, parent_logvar)
        sampled.append(x_sampled)

        p_x_node = self.p_x_node(x, parent_root, parent_logvar)
        p_x_nodes.append(p_x_node)

        means.append(parent_root)
        logvars.append(parent_logvar)

        logtis = torch.matmul(x, parent_root.T)
        layer_logits.append(logtis)

        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar = layer(parent_root, parent_logvar)
            # logtis = torch.matmul(x, parent_root.T)
            # layer_logits.append(logtis)

            x_sampled = self.sample(parent_root, parent_logvar)
            sampled.append(x_sampled)

            p_x_node = self.p_x_node(x, parent_root, parent_logvar)
            p_x_nodes.append(p_x_node)

            means.append(parent_root)
            logvars.append(parent_logvar)

            logtis = torch.matmul(x, parent_root.T)
            layer_logits.append(logtis)


        
        return x, means, logvars, sampled, p_x_nodes

            
        # loss = 0
        # for logits, mean in zip(layer_logits, means):
        #     probs = F.softmax(logits, dim=-1)
        #     # print(probs)
        #     x_preds = torch.matmul(probs, mean)
        #     loss += F.mse_loss(x_preds, x)

        # return untils.ModelOutput(loss=loss, reconstructions=x_preds, layer_outputs=means)

        

# tiny example
# very similiar to the prototypical network
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.children_mean = nn.Parameter(torch.rand(8, 784))
    
    def forward(self, x, y=None, hard=False):
        x = x.view(-1, 784)
        # x: [batch_size, 784]
        # calculate the l2 distance between x and left_mean
        # cluster_dists = torch.norm(self.children_mean - x[:, None], p=2, dim=-1)
        logits = torch.matmul(x, self.children_mean.T)
        clusters_probs = F.softmax(logits, dim=-1)
        x_hat = torch.matmul(clusters_probs, self.children_mean) # shape: batch_size, 784

        loss = F.mse_loss(x_hat, x)
        return untils.ModelOutput(loss=loss)