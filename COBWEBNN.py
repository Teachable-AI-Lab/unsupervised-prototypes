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
        # print(alpha)
        # mean_root = mean_leaf * alpha # shape: n_clusters, 2, n_hidden
        # mean_root = mean_root.reshape(-1, 2, self.n_hidden).sum(dim=1) # shape: n_clusters, n_hidden
        # print(mean_root)
        return mean_root, logvar_root
    
class CobwebNN(nn.Module):
    def __init__(self, image_shape, n_hidden, n_layers):
        super(CobwebNN, self).__init__()
        self.image_shape = image_shape
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.leaves = nn.Parameter(torch.randn(2**n_layers, n_hidden))
        self.leaves_logvar = nn.Parameter(torch.randn(2**n_layers, n_hidden))

        self.layers = nn.ModuleList(
            [CobwebNNTreeLayer(n_hidden, 2**i) for i in reversed(range(0, n_layers))]
        )

        # self.fc = nn.Linear(input_dim, 400)
        self.relu = nn.ReLU()
        # self.mu = nn.Linear(400, n_hidden)
        # self.logvar = nn.Linear(400, n_hidden)



        C, H, W = image_shape
        self.input_dim = C * H * W

        self.fc = nn.Linear(self.input_dim, 400)
        self.mu = nn.Linear(400, n_hidden)

        # self.encoder = nn.Sequential(
        #     nn.Linear(self.input_dim, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, self.n_hidden)
        # )


        # self.decoder = nn.Sequential(
        #     nn.Linear(self.n_hidden, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, self.input_dim)
        # )

        CNN_config = {
            'conv1': [C, 16, 3, 2, 1],
            'conv2': [16, 32, 3, 2, 1],
            'conv3': [32, 64, 3, 1, 0],
        }

        # CNN autoencoder
        self.encoder = nn.Sequential(
            # nn.Conv2d(C, 16, 3, stride=2, padding=1),
            nn.Conv2d(CNN_config['conv1'][0], 
                      CNN_config['conv1'][1], 
                      CNN_config['conv1'][2], 
                      stride=CNN_config['conv1'][3], 
                      padding=CNN_config['conv1'][4]),
            nn.ReLU(),
            # nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Conv2d(CNN_config['conv2'][0], 
                      CNN_config['conv2'][1], 
                      CNN_config['conv2'][2], 
                      stride=CNN_config['conv2'][3], 
                      padding=CNN_config['conv2'][4]),
            nn.ReLU(),
            # nn.Conv2d(32, 64, 3),
            nn.Conv2d(CNN_config['conv3'][0], 
                      CNN_config['conv3'][1], 
                      CNN_config['conv3'][2], 
                      stride=CNN_config['conv3'][3], 
                      padding=CNN_config['conv3'][4]),
            nn.ReLU(),
        )

        self.H_conv = H
        self.W_conv = W
        self.out_channels = CNN_config['conv3'][1]

        # infer the shape of the output of the encoder
        for CNN_layer in CNN_config.values():
            self.H_conv = (self.H_conv + CNN_layer[4] * 2 - CNN_layer[2]) // CNN_layer[3] + 1
            self.W_conv = (self.W_conv + CNN_layer[4] * 2 - CNN_layer[2]) // CNN_layer[3] + 1

        self.CNN_output_dim = self.out_channels * self.H_conv * self.W_conv

        # self.encoder_fc = nn.Linear(64 * 5 * 5, self.n_hidden)
        self.encoder_fc = nn.Linear(self.H_conv * self.W_conv * self.out_channels, self.n_hidden)

        # self.decoder_fc = nn.Linear(self.n_hidden, 64 * 5 * 5)
        self.decoder_fc = nn.Linear(self.n_hidden, self.H_conv * self.W_conv * self.out_channels)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, 3),
            nn.ConvTranspose2d(CNN_config['conv3'][1],
                               CNN_config['conv3'][0],
                               CNN_config['conv3'][2]),

            nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(CNN_config['conv2'][1],
                                CNN_config['conv2'][0],
                                CNN_config['conv2'][2],
                                stride=CNN_config['conv2'][3],
                                padding=CNN_config['conv2'][4],
                                output_padding=1),
            nn.ReLU(),
            # nn.ConvTranspose2d(16, C, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(CNN_config['conv1'][1],
                                CNN_config['conv1'][0],
                                CNN_config['conv1'][2],
                                stride=CNN_config['conv1'][3],
                                padding=CNN_config['conv1'][4],
                                output_padding=1),

            # nn.Sigmoid()
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
        logvar: [n_clusters, n_hidden]

        p(x|node) = N(x|mean, var)
        log(p(x|node)) = -0.5 * (log(2*pi) + log(var) + (x - mean)^2 / var)
        '''

        # Expand dimensions: x is [B, n_hidden] -> [B, 1, n_hidden]
        # mean and logvar are [n_clusters, n_hidden] -> [1, n_clusters, n_hidden]
        x_expanded = x.unsqueeze(1) # shape [B, 1, n_hidden]
        mean_expanded = mean.unsqueeze(0) # shape [1, n_clusters, n_hidden] 
        logvar_expanded = logvar.unsqueeze(0) # shape [1, n_clusters, n_hidden]
        var_expanded = torch.exp(logvar_expanded)        
        # Compute constant: log(2π)
        log2pi = torch.log(torch.tensor(2 * 3.141592653589793, device=x.device))
        
        # Compute log probability per dimension and sum over hidden dimensions.
        # This gives a tensor of shape [B, n_clusters]
        log_prob = -0.5 * (log2pi + logvar_expanded + ((x_expanded - mean_expanded) ** 2) / var_expanded)
        log_prob = log_prob.sum(dim=-1)
        
        return log_prob

    def kl_div(self, x_mean, x_logvar, node_mean, node_logvar):
        '''
        Calculate the KL divergence between x and the node
        x_mean: [batch_size, n_hidden]
        x_logvar: [batch_size, n_hidden]

        node_mean: [n_clusters, n_hidden]
        node_logvar: [n_clusters, n_hidden]

        return: [batch_size, n_clusters]
        '''
        # Expand dimensions to properly broadcast the inputs
        x_mean = x_mean.unsqueeze(1)
        x_logvar = x_logvar.unsqueeze(1)
        
        # node is reshaped to [1, n_clusters, n_hidden]
        node_mean = node_mean.unsqueeze(0)
        node_logvar = node_logvar.unsqueeze(0)
        # Compute KL divergence for each (batch, node) pair using the diagonal Gaussian formula:
        # KL = 0.5 * [ log(node_var/x_var) + (x_var + (x_mean - node_mean)^2) / node_var - 1 ]
        kl = 0.5 * torch.sum(
            node_logvar - x_logvar +
            torch.exp(x_logvar - node_logvar) +
            ((x_mean - node_mean) ** 2) / torch.exp(node_logvar) - 1,
            dim=-1
        )
        # print(kl)
        
        return kl


        # return kl.sum(dim=-1) # shape: [B, n_clusters]
    
    def forward(self, x, y=None, hard=None):
        # x = x.view(-1, self.input_dim)

        B = x.size(0)
        # x_mu = self.encoder_fc(self.encoder(x).view(-1, 16 * 16 * 16))
        # print(self.encoder(x).shape)
        # print(x.shape)

        # print(self.out_channels, self.H_conv, self.W_conv, self.CNN_output_dim)

        x_mu = self.encoder_fc(self.encoder(x).view(-1, self.CNN_output_dim))


        # x_mu = self.mu(self.relu(self.fc(x)))
        # x_logvar = self.logvar(self.relu(self.fc(x)))

        # B = x.size(0)
        
        parent_root = self.leaves
        parent_logvar = self.leaves_logvar
        means = []
        logvars = []
        sampled = []
        p_x_nodes = []
        p_node_x = []
        layer_logits = []
        x_preds = []

        hard = False
        alpha = 0.2
        tau = 1

        # kl_div = self.kl_div(x_mu, x_logvar, parent_root, parent_logvar)
        # distance to root
        # Eucledian distance
        # print(x_mu.shape, parent_root.shape)
        kl_div = torch.norm(x_mu.unsqueeze(1) - parent_root.unsqueeze(0), p=2, dim=-1)
        # kl_div = torch.matmul(x_mu, parent_root.T) # shape: B, n_clusters
        # kl_probs = F.softmax(-kl_div, dim=-1) # shape: B, n_clusters
        kl_probs = untils.GumbelSoftmax(-kl_div, tau=tau, alpha=alpha, hard=hard)
        p_node_x.append(kl_probs)
        # print(kl_probs.shape)
        # print(kl_probs)
        sampled_x = self.sample(parent_root, parent_logvar) # shape: n_clusters, n_hidden
        # print(sampled_x.shape)
        # weighted combination
        sampled_x = (kl_probs.unsqueeze(-1) * sampled_x).sum(dim=1) # shape: B, n_hidden

        # x_pred = self.decoder(sampled_x) # shape: B, input_dim
        x_pred = self.decoder(self.decoder_fc(sampled_x).view(-1, self.out_channels, self.H_conv, self.W_conv))

        # x_pred = self.decoder(self.decoder_fc(sampled_x).view(-1, 16, 16, 16))

        x_preds.append(x_pred)

        # means.append(parent_root)
        # logvars.append(parent_logvar)

        # layer_logits.append(torch.matmul(x, parent_root.T))

        # x_sampled = self.sample(parent_root, parent_logvar)
        # sampled.append(x_sampled)

        p_x_node = self.p_x_node(x_mu, parent_root, parent_logvar)
        p_x_node = untils.GumbelSoftmax(p_x_node, tau=tau, alpha=alpha, hard=hard)

        p_x_nodes.append(p_x_node)
        # print(p_x_node.shape)

        means.append(parent_root)
        logvars.append(parent_logvar)

        # logtis = torch.matmul(x, parent_root.T)
        # layer_logits.append(logtis)

        for i, layer in enumerate(self.layers):
            parent_root, parent_logvar = layer(parent_root, parent_logvar)
            # logtis = torch.matmul(x, parent_root.T)
            # layer_logits.append(logtis)

            # kl_div = self.kl_div(x_mu, x_logvar, parent_root, parent_logvar)
            # kl_div = torch.norm(x_mu - parent_root, p=2, dim=-1)
            kl_div = torch.norm(x_mu.unsqueeze(1) - parent_root.unsqueeze(0), p=2, dim=-1)
            # kl_div = torch.matmul(x_mu, parent_root.T) # shape: B, n_clusters
            # kl_probs = F.softmax(-kl_div, dim=-1) # shape: B, n_clusters
            kl_probs = untils.GumbelSoftmax(-kl_div, tau=tau, alpha=alpha, hard=hard)
            p_node_x.append(kl_probs)
            # print(kl_probs)
            sampled_x = self.sample(parent_root, parent_logvar) # shape: B, n_hidden
            # weighted combination
            sampled_x = (kl_probs.unsqueeze(-1) * sampled_x).sum(dim=1) # shape: B, n_hidden
            # x_pred = self.decoder(sampled_x) # shape: B, input_dim
            x_pred = self.decoder(self.decoder_fc(sampled_x).view(-1, self.out_channels, self.H_conv, self.W_conv))
            # x_pred = self.decoder(self.decoder_fc(sampled_x).view(-1, 16, 16, 16))
            x_preds.append(x_pred)


            # x_sampled = self.sample(parent_root, parent_logvar)
            # sampled.append(x_sampled)

            p_x_node = self.p_x_node(x_mu, parent_root, parent_logvar)
            # softmax this thing
            p_x_node = untils.GumbelSoftmax(p_x_node, tau=tau, alpha=alpha, hard=hard)
            # p_x_node = self.p_x_node(x_mu, parent_root, parent_logvar)
            # p_x_node = p_x_node * untils.GumbelSoftmax(p_x_node, tau=tau, alpha=alpha, hard=hard)
            p_x_nodes.append(p_x_node)

            means.append(parent_root)
            logvars.append(parent_logvar)

            # logtis = torch.matmul(x, parent_root.T)
            # layer_logits.append(logtis)


        
        return x, means, logvars, x_preds, p_x_nodes, p_node_x

            
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
    
