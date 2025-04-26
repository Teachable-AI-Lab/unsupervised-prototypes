# define several helper functions for cobweb-nn experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from math import log2  # Import log2 directly to avoid shadowing issues
from collections import defaultdict
import time # For timing initialization
import warnings # For clearer warnings
import random
import json
import os
import argparse
import torchvision.models as models
from collections import OrderedDict



def add_noise(x, noise_level=0.1, noise_type='gaussian'):
    '''
    x: (batch, C, H, W)
    noise_type: 'gaussian', 'random_mask'
    '''
    if noise_type == 'gaussian':
        noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
    elif noise_type == 'random_mask':
        # Create a random mask
        mask = torch.rand_like(x) < noise_level
        # Apply the mask to the input
        x_noisy = x * mask
    else:
        raise ValueError("Unsupported noise type. Use 'gaussian' or 'random_mask'.")
    return x_noisy

# ------------------------------
# KL Annealing Scheduler (Linear)
# ------------------------------
def linear_annealing(epoch, anneal_epochs=50):
    # Increase beta linearly until it reaches 1.0
    return min(1.0, (epoch) / anneal_epochs)

def convex_weight_decay(n_layes, _lambda):
    weights = []
    for i in range(1, n_layes+1):
        lenth = 2**i
        weight = np.exp(-_lambda * (i - 1))
        # weights.extend([weight] * lenth)
        weights.append(torch.tensor([weight] * lenth))
    # print(f"weights: {weights}")
    weights = weights[::-1]
    weights = torch.cat(weights, dim=0)
    return weights

def dkl_weight_warmup(n_layes, margin, _lambda):
    weights = []
    for i in range(0, n_layes):
        lenth = 2**i
        weight = margin * np.exp(_lambda * i)
        # weights.extend([weight] * lenth)
        weights.append(torch.tensor([weight] * lenth))
    # print(f"weights: {weights}")
    weights = weights[::-1]
    # weights = torch.tensor(weights)
    weights = torch.cat(weights, dim=0)
    return weights

class NoiseScheduler:
    def __init__(self, num_steps, start_noise=1.0, end_noise=0.0):
        self.num_steps = num_steps
        self.start_noise = start_noise
        self.end_noise = end_noise
        self.current_step = 0

    def get_noise(self):
        if self.current_step < self.num_steps:
            noise = self.start_noise + (self.end_noise - self.start_noise) * (self.current_step / self.num_steps)
        else:
            noise = self.end_noise
        return noise
        
    def step(self):
        self.current_step += 1
        
def get_loss_weights(epoch: int,
                     recon_intv: int,
                     dkl_intv: int,
                     start_first: str = 'recon') -> tuple[int, int]:
    """
    Returns binary weights (recon_w, dkl_w) for the given epoch:

    - If both intervals are zero, both weights are 1.
    - If recon_intv == 0, recon_w is always 1; schedule dkl as alternating blocks.
    - If dkl_intv   == 0, dkl_w   is always 1; schedule recon as alternating blocks.
    - Otherwise, interleave recon_intv epochs of recon and dkl_intv epochs of dkl,
      starting with start_first ('recon' or 'dkl').
    """
    # Both always-on
    if recon_intv == 0 and dkl_intv == 0:
        return 1, 1

    # Only recon always-on: schedule dkl in alternating block
    if recon_intv == 0:
        recon_w = 1
        offset = 0 if start_first == 'dkl' else dkl_intv
        cycle = 2 * dkl_intv
        pos = (epoch + offset) % cycle if cycle > 0 else 0
        dkl_w = int(pos < dkl_intv)
        return recon_w, dkl_w

    # Only dkl always-on: schedule recon in alternating block
    if dkl_intv == 0:
        dkl_w = 1
        offset = 0 if start_first == 'recon' else recon_intv
        cycle = 2 * recon_intv
        pos = (epoch + offset) % cycle if cycle > 0 else 0
        recon_w = int(pos < recon_intv)
        return recon_w, dkl_w

    # Both interleave normally
    cycle = recon_intv + dkl_intv
    pos = epoch % cycle
    if start_first == 'recon':
        recon_w = int(pos < recon_intv)
        dkl_w   = int(pos >= recon_intv)
    elif start_first == 'dkl':
        dkl_w   = int(pos < dkl_intv)
        recon_w = int(pos >= dkl_intv)
    else:
        raise ValueError("start_first must be 'recon' or 'dkl'")

    return recon_w, dkl_w

def pretrain(model, train_loader, optimizer, epochs, device):
    """
    Pretrain the model using MSE loss.
    Only the encoder and decoder are trained.
    """
    model.train()
    for epoch in tqdm(range(epochs)):
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            # z, logvar = model.encode(images)
            # normliaze z to around mean 0 and std 1
            # z = (z - z.mean()) / z.std()
            # print z range
            # print(f"z range: {z.min()} {z.max()}")
            # print(z[0])
            # h = model.encoder(images)
            # x_recon = model.decoder_raw(h)
            # x_recon = model.decode(z)
            # loss = F.mse_loss(x_recon, images)
            loss, _, _, _ = model.vae_forward(images)
            loss.backward()
            optimizer.step()
    # radnomly initialize the model.mu_c by filling it with random z
    # with torch.no_grad():
    #     # randomly select 2**model.n_layers samples from z
    #     z = z[torch.randperm(z.size(0))[:2**model.n_layers]]
    #     model.mu_c.data = z

    #     logvar = logvar[torch.randperm(logvar.size(0))[:2**model.n_layers]]
    #     model.logvar_c.data = logvar

    # print(model.mu_c)
    # print(torch.max(model.mu_c), torch.min(model.mu_c))
    # print(torch.max(model.logvar_c), torch.min(model.logvar_c))

def label_annotation(model, support_loader, n_classes, device):
    '''
    support_set: N, input_dim, The training data

    return: a n_classs x n_nodes matrix. Each column stores the class distribution of the corresponding cluster.
    '''
    model.eval()
    pcx = []
    labels = []

    # support_loader = DataLoader(suppoer_set, batch_size=512, shuffle=False)

    with torch.no_grad():
        for i, (image, label) in enumerate(support_loader):
            image = image.to(device)
            _, _, _, _, _, pcx_batch, _, _ = model(image)
            pcx.append(pcx_batch)
            labels.append(label)

    pcx = torch.cat(pcx, dim=0) # shape: (N, n_nodes)

    labels = torch.cat(labels, dim=0) # shape: (N, )

    N = pcx.shape[0]
    n_nodes = pcx.shape[1]

    # turn pcx into one-hot
    # pcx = pcx.argmax(dim=1) # shape: (N, ) # return index? Answer: yes
    # pcx = F.one_hot(pcx, num_classes=n_nodes) # shape: (N, n_classes)

    annotation = torch.zeros(n_classes, n_nodes)


    for c in range(n_classes):
        class_indices = (labels == c)

        pcx_c = pcx[class_indices] # shape: (N_c, n_nodes)
        pcx_c = pcx_c.sum(dim=0) # shape: (n_nodes, )

        annotation[c, :] = pcx_c
    
    # only look at the leaves
    # annotation = annotation[:, n_nodes // 2:]
    # print(f"annotation shape: {annotation.shape}")
    # normlize the annotation on each column
    annotation = annotation / torch.sum(annotation, dim=0, keepdim=True)

    return annotation

def basic_node_evaluation(model, annotation, query_loader, device):
    # annotation: n_classes x n_nodes
    model.eval()
    pcx = []
    labels = []

    # query_loader = DataLoader(query_set, batch_size=512, shuffle=False)

    with torch.no_grad():
        for i, (image, label) in enumerate(query_loader):
            image = image.to(device)
            _, _, _, _, _, pcx_batch, _, _ = model(image)
            pcx.append(pcx_batch.detach().cpu())
            labels.append(label)

    pcx = torch.cat(pcx, dim=0) # shape: (N, n_nodes)
    labels = torch.cat(labels, dim=0) # shape: (N, )

    # pcx = pcx[:, pcx.shape[1] // 2] # only look at the leaves

    # find argmax of pcx
    # pred = pcx.argmax(dim=1) # shape: (N, ) # return index? Answer: yes
    # print(f"pred shape: {pred[:10]}")
    # index the columns of annotation 
    # pred = annotation[:, pred] # shape: (n_classes, N)
    # pred = pred.T # shape: (N, n_classes)

    # weight the columns of annotation by pcx
    pred = pcx @ annotation.T # shape: (N, n_classes)
    # print(f"pred shape: {pred.shape}")

    pred = pred.argmax(dim=1) # shape: (N, ) # return index? Answer: yes

    # compute accuracy
    correct = (pred == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    # print(f"Accuracy: {acc:.4f}")
    return acc

    # print(pred[0])

def testing_few_shot(model, fs_loader, device):
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for test_episode in tqdm(fs_loader):
            support_img = test_episode['support_images'].to(device)
            support_labels = test_episode['support_labels']#.to(device)
            query_img = test_episode['query_images'].to(device)
            query_labels = test_episode['query_labels']#.to(device)

            # annotate the model with support set
            _, _, _, _, _, pcx_support, _, _ = model(support_img)
            pcx_support = pcx_support.detach().cpu()
            N = pcx_support.shape[0]
            n_nodes = pcx_support.shape[1]
            n_classes = len(torch.unique(support_labels))
            annotation = torch.zeros(n_classes, n_nodes)
            # print(f"annotation shape: {annotation.shape}")
            # break

            for c in range(n_classes):
                class_indices = (support_labels == c)
                # print(class_indices)

                pcx_support_c = pcx_support[class_indices] # shape: (N_c, n_nodes)
                pcx_support_c = pcx_support_c.sum(dim=0) # shape: (n_nodes, )

                annotation[c, :] = pcx_support_c
            # break
            # normlize the annotation on each column
            # first check if any column sums to 0
            annotation = annotation / torch.sum(annotation + 1e-10, dim=0, keepdim=True)
            # print(f"annotation shape: {annotation}")
            # print(torch.sum(annotation, dim=0, keepdim=True).shape)
            # print(f"annotation {annotation.shape}")
            # break

            # test the model with query set
            _, _, _, _, _, pcx_query, _, _ = model(query_img)
            pcx_query = pcx_query.detach().cpu()
            # print(f"pcx_query shape: {pcx_query}")
            # print(f"pcx_query {pcx_query}")
            # print("NaNs in pcx_query:", torch.isnan(pcx_query).any())
            # print("NaNs in annotation:", torch.isnan(annotation).any())

            pred = pcx_query @ annotation.T # shape: (N, n_classes)
            # print(pred)
            pred = pred.argmax(dim=1) # shape: (N, ) # return index? Answer: yes
            # print(f"pred {pred}")
            # break

            correct = (pred == query_labels).sum().item()
            total_correct += correct
            total_samples += query_labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Few-shot accuracy: {accuracy:.4f}")
    return accuracy
    


def linear_probing(model, n_classes, train_loader, test_loader, lr, epochs, device):
    """
    Train a linear classifier on top of the frozen encoder.
    """
    classifier = nn.Linear(model.latent_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    best_acc = 0

    for epoch in range(epochs):
        classifier.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                mu, logvar = model.encode(images)
                z = model.reparameterize(mu, logvar)
            logits = classifier(z)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        # Evaluate the classifier on the test set
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                mu, logvar = model.encode(images)
                z = model.reparameterize(mu, logvar)
                logits = classifier(z)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
    return best_acc

def get_latent(model, test_loader, device):
    all_latent = []
    all_labels = []
    all_pcx = []
    pis = []
    centroid_list = []
    mu_c = None
    for i, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data = data.to(device)
            mu, logvar = model.encode(data)
            latent = model.reparameterize(mu, logvar)
            all_latent.append(latent.detach().cpu())
            all_labels.append(target)

            _, recon_loss, kl1, kl2, H, pcx, pi, _ = model(data)
            all_pcx.append(pcx.detach().cpu())
            pis = pi.detach().cpu().numpy()
            # pis.append(pi)
            if i == 0:
                break
    with torch.no_grad():
        mu_c = model.gmm_params()[1]
        centroids = model.decode(mu_c)
        # centroids = centroids.view(-1, shape[0], shape[1], shape[2]).cpu().numpy()
        # break it into a list
        for j in range(model.n_layers+1):
            # pop the last one
            # then pop last two
            # then pop last four
            # then pop last eight
            # follows the pattern of 2^n
            num_to_pop = 2 ** j  # This will be 1, 2, 4, 8, ... for layers 0, 1, 2, 3, etc.
            layer_centroids = centroids[-num_to_pop:]
            centroid_list.append(layer_centroids)
            centroids = centroids[:-num_to_pop]
                
                
    all_latent = torch.cat(all_latent, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_latent = all_latent.detach().cpu().numpy()
    # concate with mu_c
    all_latent = np.concatenate((all_latent, mu_c.detach().cpu().numpy()), axis=0)
    all_labels = all_labels.cpu().numpy()
    pcx = torch.cat(all_pcx, dim=0).detach().cpu().numpy() # batch_size, n_classes

    return all_latent, all_labels, pcx, pis, centroid_list, H


def plot_tsne(all_latent, all_labels):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(all_latent)
    # plt.figure(figsize=(5, 5))
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(tsne_results[:len(all_labels), 0], tsne_results[:len(all_labels), 1], c=all_labels, cmap='tab10', s=5)
    plt.colorbar()
    plt.scatter(tsne_results[len(all_labels):, 0], tsne_results[len(all_labels):, 1], c='black', s=10, marker='x')

    # return the figiure
    return fig

def plot_qcx(pcx):
    fig = plt.figure(figsize=(5, 5))
    plt.plot(pcx.T, c='blue', alpha=0.01)
    return fig

def plot_pi(pis):
    fig = plt.figure(figsize=(5, 5))
    plt.plot(pis.T, c='red', alpha=0.5)
    return fig

def plot_entropy(H):
    num_layers = len(H)
    colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
    fig = plt.figure(figsize=(5, 5))
    for i, entropy in enumerate(H):
        entropy = entropy.cpu().numpy()
        plt.scatter(np.arange(entropy.shape[0]), entropy, alpha=0.5, color=colors[i], label=f'Layer {len(H) - i - 1}')
    plt.legend()
    return fig

def plot_centroids(centroids_list, layer):
    centroids = centroids_list[layer].cpu().numpy()
    # print(f"centroids shape: {centroids.shape}")
    # centroids = centroids.view(-1, *
    # plot the centroids    
    num_centroids = centroids.shape[0]
    # Create a figure with subplots arranged in one row
    fig, axes = plt.subplots(1, num_centroids, figsize=(num_centroids, 1))
    
    # In case there is only one centroid, convert axes to a list for easy iteration
    if num_centroids == 1:
        axes = [axes]
    # Plot each centroid image in its corresponding subplot
    for i, ax in enumerate(axes):
        img = centroids[i]
        # For grayscale images (1 channel), squeeze out the channel dimension
        if centroids.shape[1] == 1:
            img = img.squeeze(0)
            ax.imshow(img, cmap='gray')
        else:
            # For color images, convert the image from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            ax.imshow(img)
        ax.axis('off')  # Remove axis ticks/labels

    # Remove any spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    return fig


def plot_generated_examples(model, layer, n_examples, device):
    with torch.no_grad():
        _, mu_c, logvar_c, _, _, _ = model.gmm_params()
        mu_c_list = []
        logvar_c_list = []
        # centroids = centroids.view(-1, shape[0], shape[1], shape[2]).cpu().numpy()
        # break it into a list
        for j in range(model.n_layers+1):
            num_to_pop = 2 ** j  # This will be 1, 2, 4, 8, ... for layers 0, 1, 2, 3, etc.
            layer_mu_c = mu_c[-num_to_pop:]
            mu_c_list.append(layer_mu_c)
            mu_c = mu_c[:-num_to_pop]

            layer_logvar_c = logvar_c[-num_to_pop:]
            logvar_c_list.append(layer_logvar_c)
            logvar_c = logvar_c[:-num_to_pop]

        # print(len(mu_c_list))

        mu_c = mu_c_list[layer].unsqueeze(1).expand(-1, n_examples, -1).reshape(-1, mu_c.shape[-1]) # shape: (n_clusters * n_examples, n_hidden)
        logvar_c = logvar_c_list[layer].unsqueeze(1).expand(-1, n_examples, -1).reshape(-1, logvar_c.shape[-1]) # shape: (n_clusters * n_examples, n_hidden)
        eps = torch.randn_like(mu_c)  # shape: (n_clusters * n_examples, n_hidden) 
        # make the noise smaller
        eps = eps
        z = mu_c + torch.exp(logvar_c / 2) * eps  # shape: (n_clusters * n_examples, n_hidden)
        x = model.decode(z.to(device))  # shape: (n_clusters * n_examples, C, H, W)
        x = x.view(-1, n_examples, *x.shape[1:]).detach().cpu().numpy()  # shape: (n_clusters, n_examples, C, H, W)

        n_clusters = x.shape[0]

        # generate small noise for each cluster for each example
        # Create subplots: n_clusters rows and n_examples columns.
        fig, axes = plt.subplots(n_clusters, n_examples, figsize=(n_examples, n_clusters))
        # Force the axes array to be 2D.
        if n_clusters == 1 and n_examples == 1:
            axes = np.array([[axes]])
        elif n_clusters == 1:
            axes = np.expand_dims(axes, axis=0)
        elif n_examples == 1:
            axes = np.expand_dims(axes, axis=1)
            
        # Loop over clusters and examples to display each generated image.
        for i in range(n_clusters):
            for j in range(n_examples):
                ax = axes[i, j]
                img = x[i, j]
                # If the image is grayscale (1 channel), remove the extra dimension.
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                    ax.imshow(img, cmap='gray')
                else:
                    # For color images, convert from (C, H, W) to (H, W, C)
                    # img = np.transpose(img, (1, 2, 0))
                    # print(f"img shape: {img.shape}")
                    # print(f"img shape: {img.shape}")
                    ax.imshow(img.transpose(1, 2, 0))
                ax.axis("off")  # Remove axis ticks and labels
        
        # Remove spacing between subplots.
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
        return fig

def plot_dataset_examples(model, all_latents, pcx, layer, n_examples, device):
    n_clusters = pcx.shape[1]
    top_latents = []

    for c in range(n_clusters):
        pcx = torch.tensor(pcx, device=device)
        # For cluster c, get the top_k indices from the batch with highest pcx values.
        top_values, top_indices = torch.topk(pcx[:, c], n_examples, dim=0)
        # Gather the corresponding latent vectors, shape: (top_k, n_hidden)
        # Note: all_latents is assumed to be a tensor of shape (n_samples, n_hidden)
        # all_latents is a numpy array, so we need to convert it to a tensor
        all_latents = torch.tensor(all_latents, device=device)
        cluster_latents = all_latents[top_indices]
        top_latents.append(cluster_latents.unsqueeze(0))  # unsqueeze to add cluster dimension

    # Concatenate along the cluster dimension: resulting shape is (n_clusters, top_k, n_hidden)
    output = torch.cat(top_latents, dim=0)
    # Reshape to (n_clusters * top_k, n_hidden)
    output = output.view(-1, output.shape[-1])
    with torch.no_grad():
        # Decode the latent vectors to get the generated images
        generated_images = model.decode(output)
        # Reshape to (n_clusters, top_k, C, H, W)
    generated_images = generated_images.view(len(top_latents), n_examples, *generated_images.shape[1:])
    # Move to CPU and convert to numpy
    generated_images = generated_images.cpu().numpy()
    # print(f"generated_images shape: {generated_images.shape}")
    n = generated_images.shape[0]  # total number of nodes: should equal 2^L - 1
    L = int(np.log2(n + 1))  # calculate the number of layers L
    start = 2**L - 2**(layer + 1)
    end = start + 2**layer
    generated_images = generated_images[start:end, ...]
    # print(f"generated_images shape: {generated_images.shape}")
    # print()


    # Create subplots: n_clusters rows and n_examples columns.
    fig, axes = plt.subplots(end-start, n_examples, figsize=(n_examples, end-start))
    # Force the axes array to be 2D.
    if end-start == 1 and n_examples == 1:
        axes = np.array([[axes]])
    elif end-start == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_examples == 1:
        axes = np.expand_dims(axes, axis=1)
        
    # Loop over clusters and examples to display each generated image.
    for i in range(end-start):
        for j in range(n_examples):
            ax = axes[i, j]
            img = generated_images[i, j]
            # If the image is grayscale (1 channel), remove the extra dimension.
            if img.shape[0] == 1:
                img = img.squeeze(0)
                ax.imshow(img, cmap='gray')
            else:
                # For color images, convert from (C, H, W) to (H, W, C)
                # img = np.transpose(img, (1, 2, 0))
                # print(f"img shape: {img.shape}")
                # print(f"img shape: {img.shape}")
                ax.imshow(img.transpose(1, 2, 0))
            ax.axis("off")  # Remove axis ticks and labels
    
    # Remove spacing between subplots.
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    return fig



def GumbelSoftmax(logits, tau=1, alpha=1, hard=False, dim=-1):
    def _gumbel():
        gumbel = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbel).sum() or torch.isinf(gumbel).sum():
            gumbel = _gumbel()
        return gumbel
    
    gumbel = _gumbel()
    # print(logits.argmax(dim=-1))
    gumbel = (logits + gumbel * alpha) / tau
    # print(gumbel.argmax(dim=-1))

    # print(f"number of mathces: {torch.sum(logits.argmax(dim=-1) == gumbel.argmax(dim=-1))}")
    # y_soft = F.softmax(gumbel, dim=dim)
    y_soft_log = gumbel - gumbel.logsumexp(dim, keepdim=True)
    y_soft = y_soft_log.exp()
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret, y_soft_log


def get_data_loader(dataset, batch_size, normalize, 
                    N_WAY_TEST=5, K_SHOT_TEST=1, N_QUERY_TEST=15, N_TEST_EPISODES=600,
                    train_subset=1):
    # Always download if not present
    download = True
    # Convert dataset name to lowercase for flexible input
    dataset = dataset.lower()

    if dataset == 'cifar-10':
        dataset_class = datasets.CIFAR10
        data_dir = 'data/CIFAR10'
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

    elif dataset == 'cifar-100':
        dataset_class = datasets.CIFAR100
        data_dir = 'data/CIFAR100'
        if normalize:
            # Common normalization values for CIFAR-100
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

    elif dataset == 'mnist':
        dataset_class = datasets.MNIST
        data_dir = 'data/MNIST'
        if normalize:
            # Common normalization parameters for MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
    elif dataset == 'fashion-mnist':
        dataset_class = datasets.FashionMNIST
        data_dir = 'data/FashionMNIST'
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
    elif dataset == 'svhn':
        dataset_class = datasets.SVHN
        data_dir = 'data/SVHN'
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
    # elif dataset == 'stl10':
    #     dataset_class = datasets.STL10
    #     data_dir = 'data/STL10'
    #     if normalize:
    #         transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #         ])
    #     else:
    #         transform = transforms.Compose([
    #             transforms.ToTensor()
    #         ])
    elif dataset == 'stl-10' or dataset == 'stl-10-eval':
        dataset_class = datasets.STL10
        data_dir = 'data/STL10'
        if normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
    elif dataset == 'omniglot':
        dataset_class = datasets.Omniglot
        data_dir = 'data/OMNIGLOT'
        if normalize:
            transform = transforms.Compose([
                # resize to 28x28
                transforms.Resize(28),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(28),
                transforms.ToTensor()
            ])
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar-10', 'cifar-100', or 'mnist'.")

    if dataset == 'omniglot':
        train_set = dataset_class(root=data_dir, download=download, transform=transform)
        test_set = dataset_class(root=data_dir, background=False, download=download, transform=transform)
        test_classes = sorted(list(set(label for _, label in test_set)))
        test_set = EpisodeDataset(
            dataset=test_set,
            class_list=test_classes,
            n_way=N_WAY_TEST,
            k_shot=K_SHOT_TEST,
            n_query=N_QUERY_TEST,
            n_episodes_per_epoch=N_TEST_EPISODES # Generate enough episodes for evaluation
        )
        # radnomly split the dataset into 2 parts: 80% for training and 20% for testing

    elif dataset == 'svhn':
        # SVHN has train and test splits
        train_set = dataset_class(root=data_dir, split='train', download=download, transform=transform)
        test_set = dataset_class(root=data_dir, split='test', download=download, transform=transform)
        # radnomly split the dataset into 2 parts: 80% for training and 20% for testing

    elif dataset == 'stl-10':
        # STL-10 has train and test splits
        train_set = dataset_class(root=data_dir, split='unlabeled', download=download, transform=transform)
        test_set = dataset_class(root=data_dir, split='test', download=download, transform=transform)
        # radnomly split the dataset into 2 parts: 80% for training and 20% for testing

    elif dataset == 'stl-10-eval':
        train_set = dataset_class(root=data_dir, split='train', download=download, transform=transform)
        test_set = dataset_class(root=data_dir, split='test', download=download, transform=transform)
    else:
    # Create the training and testing datasets
        train_set = dataset_class(root=data_dir, train=True, download=download, transform=transform)
        test_set = dataset_class(root=data_dir, train=False, download=download, transform=transform)

    # Create DataLoaders for training and testing
    # randomly select train_subset of the training set
    if train_subset < 1:
        indices = np.random.choice(len(train_set), int(len(train_set) * train_subset), replace=False)
        train_set = torch.utils.data.Subset(train_set, indices)
        print(f"Using {len(train_set)} samples for training.")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    if dataset == 'omniglot':
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=None, # EpisodeDataset yields full episodes as batches
            num_workers=0 # Often safer for IterableDatasets, adjust if needed
        )
    else:
        test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

    return train_loader, test_loader, train_set, test_set

def load_config_from_json(config_path):
    """Loads configuration parameters from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {config_path}: {e}")

    # Convert the dictionary into a Namespace object for dot notation access
    config_args = argparse.Namespace(**config_dict)

    # --- Specific Type Conversions ---
    # Convert 'dec_hidden_dim' from list (JSON array) back to tuple
    if hasattr(config_args, 'dec_hidden_dim') and isinstance(config_args.dec_hidden_dim, list):
        config_args.dec_hidden_dim = tuple(config_args.dec_hidden_dim)
        # print("Converted 'dec_hidden_dim' to tuple.") # Optional: for debugging

    # Add any other necessary type conversions here if JSON loading
    # doesn't produce the exact type required by your downstream code.

    return config_args

######## Episodic Dataset ########
class EpisodeDataset(IterableDataset):
    """
    Generates batches for N-way K-shot classification tasks using PyTorch IterableDataset.

    Each batch corresponds to a single few-shot task (episode).
    It samples N classes, then K support examples and Q query examples for each class.

    Designed to work with a standard PyTorch Dataset that returns (image, label) tuples.
    """
    def __init__(self, dataset, class_list, n_way, k_shot, n_query, n_episodes_per_epoch):
        """
        Initializes the EpisodeDataset.

        Args:
            dataset (torch.utils.data.Dataset): The underlying dataset.
                                                 Expected: dataset[i] returns (image_tensor, class_label).
                                                 Optimization: Can optionally have `dataset.class_indices`
                                                 (dict: class -> [indices]) and `dataset.get_label(index)`
                                                 for faster initialization.
            class_list (iterable): A list or set of unique class labels to sample episodes from
                                   (e.g., training classes or test classes for this specific sampler).
            n_way (int): Number of classes per episode (N).
            k_shot (int): Number of support examples per class (K).
            n_query (int): Number of query examples per class (Q).
            n_episodes_per_epoch (int): How many episodes (batches) to generate per epoch.
        """
        super().__init__()

        # --- 1. Input Validation ---
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise TypeError("`dataset` must be a PyTorch Dataset object.")
        if not hasattr(class_list, '__iter__'):
            raise TypeError("`class_list` must be an iterable (list, set, etc.).")
        if not (isinstance(n_way, int) and n_way > 0):
            raise ValueError("`n_way` must be a positive integer.")
        if not (isinstance(k_shot, int) and k_shot > 0):
            raise ValueError("`k_shot` must be a positive integer.")
        if not (isinstance(n_query, int) and n_query >= 0): # Query set can be empty
            raise ValueError("`n_query` must be a non-negative integer.")
        if not (isinstance(n_episodes_per_epoch, int) and n_episodes_per_epoch > 0):
            raise ValueError("`n_episodes_per_epoch` must be a positive integer.")

        self.dataset = dataset
        # Store unique, sorted list of classes relevant for this sampler
        self.available_classes = sorted(list(set(class_list)))
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes_per_epoch = n_episodes_per_epoch
        self.samples_per_class_needed = self.k_shot + self.n_query

        if self.samples_per_class_needed <= 0:
             raise ValueError("k_shot + n_query must be > 0")

        # --- 2. Indexing: Map classes to their sample indices ---
        start_time = time.time()
        print("Initializing EpisodeDataset: Indexing dataset by class...")
        # This dictionary will store: {class_label: [list of indices in self.dataset]}
        self.class_to_indices = defaultdict(list)
        self._build_class_indices() # Helper method for clarity
        indexing_time = time.time() - start_time
        print(f"-> Indexing complete in {indexing_time:.2f} seconds.")
        print(f"-> Found {len(self.class_to_indices)} classes with enough samples.")

        # --- 3. Final Validation ---
        # Update available_classes based on indexing results
        self.available_classes = sorted(list(self.class_to_indices.keys()))

        if len(self.available_classes) < self.n_way:
            raise ValueError(
                f"Not enough classes with sufficient samples found. "
                f"Need {self.n_way} classes for N-way sampling, but only found "
                f"{len(self.available_classes)} classes with at least "
                f"{self.samples_per_class_needed} samples in the provided class_list."
            )

        print(f"EpisodeDataset ready: Using {len(self.available_classes)} classes for {self.n_way}-way {self.k_shot}-shot tasks "
              f"({self.n_query} query samples/class).")
        print(f"Generating {self.n_episodes_per_epoch} episodes per epoch.")

    def _build_class_indices(self):
        """
        Populates self.class_to_indices by mapping class labels to their
        corresponding indices in the underlying dataset.
        Filters classes based on the required number of samples (k_shot + n_query).
        """
        valid_class_count = 0

        # OPTIMIZATION: Check if the dataset object already provides pre-computed indices
        if hasattr(self.dataset, 'class_indices') and isinstance(self.dataset.class_indices, dict):
            print("--> Attempting to use precomputed `class_indices` from dataset.")
            found_precomputed = True
            # Iterate through the classes relevant for *this* sampler instance
            for cls_label in self.available_classes:
                if cls_label in self.dataset.class_indices:
                    indices = self.dataset.class_indices[cls_label]
                    # Check if this class has enough samples
                    if len(indices) >= self.samples_per_class_needed:
                        self.class_to_indices[cls_label] = indices
                        valid_class_count += 1
                    else:
                        warnings.warn(f"Class '{cls_label}' found in precomputed indices but has only "
                                      f"{len(indices)} samples (need {self.samples_per_class_needed}). Skipping.")
                else:
                    warnings.warn(f"Class '{cls_label}' from class_list not found in dataset.class_indices.")
            if valid_class_count > 0:
                 print(f"--> Successfully used precomputed indices for {valid_class_count} classes.")
                 return # Successfully used precomputed indices

            else:
                 print("--> Precomputed indices found but none were suitable/matched class_list. Falling back to manual indexing.")


        # FALLBACK: Manually iterate through the dataset if precomputed indices aren't suitable/available
        print("--> Manually indexing dataset by class (can be slow for large datasets)...")
        has_get_label = hasattr(self.dataset, 'get_label') and callable(self.dataset.get_label)
        if not has_get_label:
            print("    (Dataset lacks `get_label` method, will load items via __getitem__ for indexing)")

        temp_indices = defaultdict(list)
        dataset_len = len(self.dataset)
        class_list_set = set(self.available_classes) # Use set for faster lookups

        for i in range(dataset_len):
            try:
                # Get label: Use get_label if available (faster), otherwise load item
                if has_get_label:
                    label = self.dataset.get_label(i)
                else:
                    _, label = self.dataset[i] # Slower: loads image data too

                # Ensure label is a hashable type (e.g., int, string)
                if isinstance(label, torch.Tensor): label = label.item()
                elif isinstance(label, np.ndarray): label = label.item()

                # Store index if the label is in the list for this sampler
                if label in class_list_set:
                    temp_indices[label].append(i)

                # Optional: Progress indicator for very large datasets
                # if (i + 1) % 20000 == 0: print(f"    Indexed {i+1}/{dataset_len}...")

            except Exception as e:
                warnings.warn(f"Could not process index {i} during manual indexing. Error: {e}")
                continue

        # Filter the manually collected indices based on sample count
        for cls_label, indices in temp_indices.items():
            if len(indices) >= self.samples_per_class_needed:
                self.class_to_indices[cls_label] = indices
                valid_class_count += 1
            else:
                 warnings.warn(f"Class '{cls_label}' found via manual indexing but has only "
                               f"{len(indices)} samples (need {self.samples_per_class_needed}). Skipping.")
        print(f"--> Manual indexing complete. Found {valid_class_count} valid classes.")


    def __iter__(self):
        """
        Yields episode batches for one epoch.
        """
        episode_count = 0
        while episode_count < self.n_episodes_per_epoch:

            # --- 1. Sample N distinct classes for the episode ---
            try:
                # Randomly select N classes from the list of classes that have enough samples
                episode_class_labels = random.sample(self.available_classes, self.n_way)
            except ValueError:
                # This should only happen if self.available_classes < self.n_way,
                # which is checked in __init__, but included as a safeguard.
                warnings.warn(f"Could not sample {self.n_way} classes from {len(self.available_classes)} available classes. Stopping iteration.")
                break # Stop the generator

            support_indices = []
            query_indices = []
            support_local_labels_list = []
            query_local_labels_list = []

            # --- 2. Sample K support and Q query examples for each chosen class ---
            sampling_successful = True
            for local_label_idx, global_class_label in enumerate(episode_class_labels):
                all_indices_for_class = self.class_to_indices[global_class_label]
                num_available = len(all_indices_for_class)

                # Double-check sample count (should be guaranteed by __init__)
                if num_available < self.samples_per_class_needed:
                     warnings.warn(f"Class '{global_class_label}' unexpectedly found with insufficient samples ({num_available}) during iteration. Skipping episode.")
                     sampling_successful = False
                     break # Skip this episode

                # Efficiently sample K+Q indices without replacement
                # torch.randperm is faster than random.sample for integer ranges
                shuffled_local_indices = torch.randperm(num_available)
                selected_local_indices = shuffled_local_indices[:self.samples_per_class_needed]
                # Map these back to the actual dataset indices
                selected_global_indices = [all_indices_for_class[i.item()] for i in selected_local_indices]

                # Split into support and query sets
                class_support_indices = selected_global_indices[:self.k_shot]
                class_query_indices = selected_global_indices[self.k_shot:]

                support_indices.extend(class_support_indices)
                query_indices.extend(class_query_indices)

                # Assign the *local* episode label (0 to N-1) corresponding to this class
                support_local_labels_list.extend([local_label_idx] * self.k_shot)
                query_local_labels_list.extend([local_label_idx] * self.n_query)

            # If sampling failed for any class in this episode, skip to the next iteration
            if not sampling_successful:
                continue

            # --- 3. Fetch the actual data (images) using the sampled indices ---
            try:
                # This step involves accessing self.dataset[idx], which might be slow if loading from disk.
                support_items = [self.dataset[idx] for idx in support_indices] # List of (img, label) tuples
                query_items = [self.dataset[idx] for idx in query_indices]     # List of (img, label) tuples
            except Exception as e:
                warnings.warn(f"Error fetching data for episode (indices: S={support_indices}, Q={query_indices}). Error: {e}. Skipping episode.")
                continue # Skip this episode

            # --- 4. Format the batch: Stack images and create label tensors ---
            try:
                # Assuming the first element of the item tuple is the image tensor
                support_images = torch.stack([item[0] for item in support_items]) # Shape: (N*K, C, H, W)
                query_images = torch.stack([item[0] for item in query_items])     # Shape: (N*Q, C, H, W)

                # Convert the local label lists (0 to N-1) to tensors
                support_local_labels = torch.tensor(support_local_labels_list, dtype=torch.long) # Shape: (N*K)
                query_local_labels = torch.tensor(query_local_labels_list, dtype=torch.long)     # Shape: (N*Q)

            except Exception as e:
                 warnings.warn(f"Error stacking tensors or creating label tensors. Error: {e}. Skipping episode.")
                 # You might want to inspect item[0].shape and type here if errors occur
                 continue # Skip this episode

            # --- 5. Yield the complete episode batch ---
            yield {
                'support_images': support_images,       # Images for the support set
                'support_labels': support_local_labels, # Corresponding local labels (0 to N-1)
                'query_images': query_images,           # Images for the query set
                'query_labels': query_local_labels,     # Corresponding local labels (0 to N-1)
                'episode_classes': episode_class_labels # Original class labels used in this episode (optional)
            }
            episode_count += 1 # Increment count only after successfully yielding an episode


    def __len__(self):
        """
        Returns the number of episodes intended to be generated per epoch.
        Note: For IterableDataset, this is often informational. The iteration stops
              based on the logic within __iter__.
        """
        return self.n_episodes_per_epoch
    

def get_pretrained(model_name):
    checkpoint_path = '/nethome/zwang910/file_storage/nips-2025/deep-taxon/pretrained/'
    model_name = model_name
    model = models.resnet50(pretrained=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- 2. Load the Checkpoint File ---
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Load onto CPU first to make key inspection and processing easier
    checkpoint = torch.load(checkpoint_path + model_name, map_location=device)

    # --- 3. Extract and Process the State Dictionary ---
    if 'state_dict' not in checkpoint:
        # If 'state_dict' is not found, maybe the checkpoint *is* the state_dict
        if isinstance(checkpoint, dict):
            print("Warning: Checkpoint does not contain 'state_dict' key. Assuming the loaded object IS the state_dict.")
            original_state_dict = checkpoint
        else:
            raise KeyError(f"Checkpoint does not seem to be a dictionary or contain the key 'state_dict'. Found type: {type(checkpoint)}")
    else:
        original_state_dict = checkpoint['state_dict']
        print("Extracted state_dict from checkpoint.")

    # *** CRITICAL STEP: Determine the correct prefix to remove ***
    # Inspect the keys of your specific checkpoint file to find the correct prefix.
    # Common prefixes for SimCLR models trained with distributed training:
    # 'module.encoder.', 'module.backbone.', 'encoder.', 'backbone.'
    # Uncomment the next line to print the first few keys and check:
    # print("First 10 keys from checkpoint state_dict:", list(original_state_dict.keys())[:10])

    prefix_to_remove = "backbone." # <--- ADJUST THIS BASED ON YOUR INSPECTION!
    print(f"Attempting to remove prefix: '{prefix_to_remove}'")

    new_state_dict = OrderedDict()
    keys_matched = 0
    keys_unmatched_printed = 0
    for k, v in original_state_dict.items():
        if k.startswith(prefix_to_remove):
            # Remove the prefix to match the standard ResNet-50 key names
            name = k[len(prefix_to_remove):]
            new_state_dict[name] = v
            keys_matched += 1
        else:
            # Keep track of keys that didn't match the prefix, might be projection head etc.
            if keys_unmatched_printed < 5: # Print first few unmatched keys
                print(f"  - Key '{k}' did not match prefix, skipping for standard ResNet load.")
                keys_unmatched_printed +=1
            elif keys_unmatched_printed == 5:
                print("  - (Further unmatched keys omitted)...")
                keys_unmatched_printed += 1
            # You could choose to add non-prefixed keys if needed, but for loading
            # into standard ResNet, we usually only want the backbone weights.
            # new_state_dict[k] = v # Uncomment this line if you expect non-prefixed keys to also be loaded

    if keys_matched == 0:
        print(f"\nWARNING: No keys matched the prefix '{prefix_to_remove}'.")
        print("Please MANUALLY INSPECT the keys in your checkpoint and adjust `prefix_to_remove`.")
        print("Checkpoint keys sample:", list(original_state_dict.keys())[:10])
        # If you are sure there's no prefix, you might comment out the processing loop
        # and use: new_state_dict = original_state_dict
    else:
        print(f"Processed {keys_matched} keys by removing prefix '{prefix_to_remove}'")


    # --- 4. Load the Processed State Dictionary into the Model ---
    print("Loading processed state_dict into the ResNet-50 model...")

    # Use strict=False. This is important because:
    #  - The SimCLR state_dict usually contains only the backbone weights, lacking the final 'fc' layer of the standard ResNet.
    #  - The original SimCLR checkpoint might contain weights for a projection head (e.g., 'projector.fc1.weight') which are not in the standard ResNet.
    # `strict=False` allows loading the weights that *do* match, ignoring the missing 'fc' keys and unexpected projection head keys.
    load_result = model.load_state_dict(new_state_dict, strict=False)

    # Report missing/unexpected keys to understand what was loaded
    print("\n--- Load State Dict Results ---")
    if not load_result.missing_keys:
        print("  - No missing keys.")
    else:
        print(f"  - Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}")
        if 'fc.weight' in load_result.missing_keys or 'fc.bias' in load_result.missing_keys:
            print("     (INFO: Missing 'fc.weight'/'fc.bias' is expected as SimCLR checkpoint contains backbone weights).")

    if not load_result.unexpected_keys:
        print("  - No unexpected keys.")
    else:
        print(f"  - Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}")
        print("     (INFO: Unexpected keys might correspond to SimCLR projection head layers not present in standard ResNet).")
    print("--- End Load State Dict Results ---")


    # --- Final Steps ---
    # Move the model to the appropriate device
    model.to(device)

    # Set the model to evaluation mode (disables dropout, uses running stats for BatchNorm)
    model.eval()
    model.fc = torch.nn.Identity()  # Remove the final classification layer

    print(f"\nSuccessfully loaded weights into model = models.resnet50(pretrained=False/weights=None).")
    print(f"Model is on device '{device}' and in evaluation mode.")

    # return model
    return model

def compute_dendrogram_purity(model, dataloader, label_annotation_matrix, device):
    """
    Compute dendrogram purity using a Monte Carlo approximation.

    Parameters:
        model: DeepTaxonNet instance
        dataloader: DataLoader for evaluation
        label_annotation_matrix: Tensor of shape (n_classes, n_nodes)
        device: torch.device
    
    Returns:
        float: Approximate dendrogram purity
    """
    from itertools import combinations
    import random

    model.eval()
    pcx_list, label_list = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, _, _, _, _, pcx_batch, _, _ = model(x)
            pcx_list.append(pcx_batch.cpu())
            label_list.append(y)

    pcx = torch.cat(pcx_list, dim=0)  # (N, n_nodes)
    labels = torch.cat(label_list, dim=0)  # (N,)
    n_nodes = pcx.shape[1]

    def lca_index(idx1, idx2):
        """Returns the least common ancestor index in a binary heap."""
        while idx1 != idx2:
            if idx1 > idx2:
                idx1 = (idx1 - 1) // 2
            else:
                idx2 = (idx2 - 1) // 2
        return idx1

    def descendant_leaves(lca_idx):
        """Return all descendant leaf indices under a given LCA index."""
        # Assumes a complete binary tree; leaves are in the bottom layer
        leaves = []
        queue = [lca_idx]
        while queue:
            node = queue.pop(0)
            left = 2 * node + 1
            right = 2 * node + 2
            if left >= n_nodes:
                leaves.append(node)
            else:
                queue.extend([left, right])
        return leaves

    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[int(label)].append(i)

    sample_pairs = []
    for indices in label_to_indices.values():
        if len(indices) >= 2:
            sampled = random.sample(list(combinations(indices, 2)), min(100, len(indices) * (len(indices) - 1) // 2))
            sample_pairs.extend(sampled)

    purity_sum = 0
    for i, j in sample_pairs:
        node_i = pcx[i].argmax().item()
        node_j = pcx[j].argmax().item()
        lca = lca_index(node_i, node_j)
        desc = descendant_leaves(lca)
        label = labels[i].item()
        count_label = sum(label_annotation_matrix[label][leaf].item() for leaf in desc)
        total = sum(label_annotation_matrix[:, leaf].sum().item() for leaf in desc)
        purity = count_label / total if total > 0 else 0
        purity_sum += purity

    return purity_sum / len(sample_pairs)


from collections import defaultdict

def compute_leaf_purity(model, dataloader, device='cuda'):
    """
    Computes the soft leaf purity for GMMDeepTaxonNet.
    Uses soft cluster assignments (p(c|x)) and true labels instead of hard assignments.
    For every sample x  we obtain the soft cluster probabilities p(c | x)
    For each leaf node, we compute the purity as the maximum soft assignment
    to the true class label divided by the total weight of that leaf.

    The overall leaf purity is the weighted average of the per-leaf purities.

    Args:
        model: GMMDeepTaxonNet model with a forward() method returning p(c|x)
        dataloader: DataLoader with (x, y) batches from test set
        device: 'cuda' or 'cpu'

    Returns:
        overall_leaf_purity: float
        per_leaf_purities: dict of {leaf_idx: (purity, total_weight)}
    """
    model.eval()
    model.to(device)

    # These collect all p(c|x) rows and the corresponding ground-truth labels
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)

             # Handle the fact that different forward() variants return different outputs. We only care about pcx
            pcx = next(
                t for t in output
                if torch.is_tensor(t) and t.dim() == 2 and t.size(0) == x_batch.size(0)
            )

            all_probs.append(pcx.cpu())
            all_labels.append(y_batch.cpu())


    # Stack everything: shapes → (N, L) and (N,)
    all_probs = torch.cat(all_probs, dim=0)     # (N, L)
    all_labels = torch.cat(all_labels, dim=0)   # (N,)
    num_leaves = all_probs.shape[1]

    # Build soft class histograms per leaf with  dictionary
    leaf_class_weights = defaultdict(lambda: defaultdict(float))
    leaf_total_weights = defaultdict(float)

    for i in range(len(all_labels)):
        label = int(all_labels[i])
        for leaf_idx in range(num_leaves):
            weight = float(all_probs[i, leaf_idx])
            leaf_class_weights[leaf_idx][label] += weight
            leaf_total_weights[leaf_idx] += weight

    # Compute purity per leaf
    per_leaf_purities = {}
    weighted_sum = 0.0
    total_weight = sum(leaf_total_weights.values())

    for leaf_idx in range(num_leaves):
        if leaf_total_weights[leaf_idx] == 0:
            purity = 0.0
        else:
            max_class_weight = max(leaf_class_weights[leaf_idx].values(), default=0.0)
            purity = max_class_weight / leaf_total_weights[leaf_idx]

        per_leaf_purities[leaf_idx] = (purity, leaf_total_weights[leaf_idx])
        weighted_sum += purity * leaf_total_weights[leaf_idx]

    overall_leaf_purity = weighted_sum / total_weight if total_weight > 0 else 0.0
    return overall_leaf_purity, per_leaf_purities

from sklearn.metrics import normalized_mutual_info_score

def compute_nmi(model, annotation, dataloader, device):
    """
    Computes the Normalized Mutual Information (NMI) between model cluster assignments and true labels.
    
    Args:
        model: DeepTaxonNet model
        annotation: label_annotation matrix (n_classes x n_nodes)
        dataloader: DataLoader for evaluation
        device: torch.device
        
    Returns:
        float: NMI score
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, _, _, _, _, pcx_batch, _, _ = model(images)
            pcx_batch = pcx_batch.detach().cpu()
            
            # Soft prediction using annotation
            preds = pcx_batch @ annotation.T  # (batch_size, n_classes)
            preds = preds.argmax(dim=1)       # Pick the best predicted class
            
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    nmi_score = normalized_mutual_info_score(all_labels, all_preds)
    return nmi_score