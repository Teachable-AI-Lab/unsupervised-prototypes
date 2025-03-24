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
from classes.resnet_using_light_basic_block_encoder import LightEncoder, LightBasicBlockEnc
from classes.resnet_using_light_basic_block_decoder import LightDecoder, LightBasicBlockDec

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
        # print(alpha)
        # mean_root = mean_leaf * alpha # shape: n_clusters, 2, n_hidden
        # mean_root = mean_root.reshape(-1, 2, self.n_hidden).sum(dim=1) # shape: n_clusters, n_hidden
        # print(mean_root)
        return mean_root, logvar_root
    
class CobwebNN(nn.Module):
    def __init__(self, 
                 image_shape=(1,28,28), 
                 n_hidden=128, 
                 n_layers=3, 
                 disable_decoder_sigmoid=False, 
                 tau=1.0, 
                 sampling=False, 
                 layer_wise=False,
                 simple_encoder=False):
        super(CobwebNN, self).__init__()
        self.image_shape = image_shape
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.sampling = sampling
        self.layer_wise = layer_wise

        # self.leaves = nn.Parameter(torch.randn(2**n_layers, n_hidden))
        # limit = 1 / torch.sqrt(torch.tensor(n_hidden, dtype=torch.float32))
        # limit = 3 ** 0.5
        # use kaiming uniform initialization
        limit = 1 / (2 ** n_layers)
        # limit = 3 ** 0.5
        self.leaves = nn.Parameter(torch.nn.init.uniform_(torch.empty(2**n_layers, n_hidden), 
                                                          -limit, limit))
        # self.leaves_logvar = nn.Parameter(torch.randn(2**n_layers, n_hidden))
        # self.leaves_logvar = nn.Parameter(torch.zeros(2**n_layers, n_hidden))
        self.leaves_logvar = nn.Parameter(torch.nn.init.uniform_(torch.empty(2**n_layers, n_hidden), 
                                                          -limit, limit))
        self.layers = nn.ModuleList(
            [CobwebNNTreeLayer(n_hidden, 2**i) for i in reversed(range(0, n_layers))]
        )

        self.relu = nn.ReLU()

        self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2])
        # self.prototype_encoder = prototype_encoder
        self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2], disable_decoder_sigmoid=disable_decoder_sigmoid)

        if simple_encoder:
            # 
            self.encoder = nn.Sequential(
                # flatten the input
                nn.Flatten(),
                nn.Linear(self.image_shape[0] * self.image_shape[1] * self.image_shape[2], 784),
                nn.ReLU(),
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_hidden),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.n_hidden, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
                nn.ReLU(),
                nn.Linear(784, self.image_shape[0] * self.image_shape[1] * self.image_shape[2]),
                nn.Sigmoid(),
                # reshape the output
                nn.Unflatten(-1, self.image_shape)
            )

        self.tau = tau

    def sample(self, mean, logvar):
        '''
        Reparameterization trick for sampling from Gaussian
        N(μ, σ) = μ + σ * ε
        where ε ~ N(0, 1)
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def tree_wise_forward(self, x, y=None, hard=None):
        x_mu = self.encoder(x).view(x.size(0), -1)

        parent_root = self.leaves # shape: n_clusters, n_hidden
        parent_logvar = self.leaves_logvar

        means = []
        logvars = []
        p_x_nodes = []

        means.append(parent_root)
        logvars.append(parent_logvar)

        hard = False
        alpha = 0.0

        ###### TREE-WISE ######
        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar = layer(parent_root, parent_logvar)

            means.append(parent_root)
            logvars.append(parent_logvar)

        # now I collect the means and logvars of entire tree
        means_cat = torch.cat(means, dim=0) # shape: 2**n_layers-1, n_hidden
        logvars_cat = torch.cat(logvars, dim=0) # shape: 2**n_layers-1, n_hidden

        # distance
        if self.sampling:
            # calculate the log probabilities of x given the nodes
            dist = -0.5 * torch.sum(logvars_cat + ((x_mu.unsqueeze(1) - means_cat.unsqueeze(0)) ** 2) / torch.exp(logvars_cat), dim=-1)
        else:
            dist = -torch.norm(x_mu.unsqueeze(1) - means_cat.unsqueeze(0), p=2, dim=-1) # shape: B, 2**n_layers-1
        # if sampling, then I'm calculating the probabilities x given the nodes
        
        dist_probs = untils.GumbelSoftmax(dist, tau=self.tau, alpha=alpha, hard=hard) # shape: B, 2**n_layers-1
        p_node_x = dist_probs

        sampled_x = means_cat
        if self.sampling:
            sampled_x = self.sample(means_cat, logvars_cat)
        # weighted combination
        sampled_x = (dist_probs.unsqueeze(-1) * sampled_x).sum(dim=1) # shape: B, n_hidden
        x_pred = self.decoder(sampled_x.view(-1, 512, 1, 1))

        return x, means, logvars, x_pred, p_x_nodes, p_node_x, x_mu, sampled_x
    
    def layer_wise_forward(self, x, y=None, hard=None):

        x_mu = self.encoder(x).view(x.size(0), -1)
        
        parent_root = self.leaves
        # parent_root = leaves
        parent_logvar = self.leaves_logvar
        means = []
        logvars = []
        p_x_nodes = []
        p_node_x = []
        x_preds = []
        x_samples = []

        hard = False
        alpha = 0.0

        ###### LAYER-WISE ######
        # distance to root
        # Eucledian distance
        if self.sampling:
            dist = -0.5 * torch.sum(parent_logvar + ((x_mu.unsqueeze(1) - parent_root.unsqueeze(0)) ** 2) / torch.exp(parent_logvar), dim=-1)
        else:
            dist = -torch.norm(x_mu.unsqueeze(1) - parent_root.unsqueeze(0), p=2, dim=-1)
        dist_probs = untils.GumbelSoftmax(dist, tau=self.tau, alpha=alpha, hard=hard)
        p_node_x.append(dist_probs)

        sampled_x = parent_root
        if self.sampling:
            sampled_x = self.sample(parent_root, parent_logvar) # shape: n_clusters, n_hidden
        # weighted combination
        sampled_x = (dist_probs.unsqueeze(-1) * sampled_x).sum(dim=1) # shape: B, n_hidden
        x_pred = self.decoder(sampled_x.view(-1, 512, 1, 1))

        x_preds.append(x_pred)
        means.append(parent_root)
        logvars.append(parent_logvar)
        x_samples.append(sampled_x)

        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar = layer(parent_root, parent_logvar)

            if self.sampling:
                dist = -0.5 * torch.sum(parent_logvar + ((x_mu.unsqueeze(1) - parent_root.unsqueeze(0)) ** 2) / torch.exp(parent_logvar), dim=-1)
            else:
                dist = -torch.norm(x_mu.unsqueeze(1) - parent_root.unsqueeze(0), p=2, dim=-1)
            dist_probs = untils.GumbelSoftmax(dist, tau=self.tau, alpha=alpha, hard=hard)
            p_node_x.append(dist_probs)
            
            sampled_x = parent_root
            if self.sampling:
                sampled_x = self.sample(parent_root, parent_logvar) # shape: B, n_hidden
            # weighted combination
            sampled_x = (dist_probs.unsqueeze(-1) * sampled_x).sum(dim=1) # shape: B, n_hidden
            x_pred = self.decoder(sampled_x.view(-1, 512, 1, 1))

            x_preds.append(x_pred)
            means.append(parent_root)
            logvars.append(parent_logvar)
            x_samples.append(sampled_x)

        
        return x, means, logvars, x_preds, p_x_nodes, p_node_x, x_mu, x_samples

            
        # loss = 0
        # for logits, mean in zip(layer_logits, means):
        #     probs = F.softmax(logits, dim=-1)
        #     # print(probs)
        #     x_preds = torch.matmul(probs, mean)
        #     loss += F.mse_loss(x_preds, x)

        # return untils.ModelOutput(loss=loss, reconstructions=x_preds, layer_outputs=means)

    def forward(self, x, y=None, hard=False):
        if self.layer_wise:
            return self.layer_wise_forward(x, y, hard)
        else:
            return self.tree_wise_forward(x, y, hard)

        

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
    
