# define several helper functions for cobweb-nn experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, IterableDataset, Dataset
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



# def augment(x, transformations):
#     '''
#     x: tensor of shape (Batch, 3, 32, 32)
#     transformations: list of transformations
#     '''
#     x = x.clone()
#     for i in range(x.shape[0]):
#         x[i] = transformations(x[i])
#     return x

def contrastive_loss(similarity_matrix, temperature=0.5):
    """
    Compute the contrastive loss given the similarity matrix.
    """
    # Get the number of samples
    batch_size_2n = similarity_matrix.shape[0]
    n = batch_size_2n // 2
    device = similarity_matrix.device

    # Scale similarities by temperature
    logits = similarity_matrix / temperature

    # Create labels: positive pair is (i, i+N) and (i+N, i)
    # For row i (0 <= i < N), target is i+N
    # For row i+N (0 <= i < N), target is i
    labels = torch.cat([torch.arange(n) + n, torch.arange(n)], dim=0)
    labels = labels.to(device) # Ensure labels are on the same device as logits

    # --- Masking out self-similarity ---
    # Create a mask to prevent examples from being compared with themselves.
    # This is crucial because exp(sim(i,i)/temp) should not be in the denominator.
    # F.cross_entropy implicitly handles the softmax and log calculation.
    # By setting the diagonal (self-similarity) logits to a very low number (-inf),
    # we ensure they don't contribute to the denominator sum in the softmax.
    mask = torch.eye(batch_size_2n, dtype=torch.bool).to(device)
    logits = logits.masked_fill(mask, -float('inf')) # Mask diagonal

    # --- Compute Cross-Entropy Loss ---
    # F.cross_entropy computes: -log(softmax(logits))[label]
    # which is equivalent to the NT-Xent loss formula when averaged over the batch.
    # Logits shape: (batch_size_2n, batch_size_2n) -> (Current Sample, All Other Samples)
    # Labels shape: (batch_size_2n) -> Index of the positive sample for each current sample
    loss = F.cross_entropy(logits, labels, reduction='mean')

    return loss

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
            _, _, _, _, _, pcx_batch, _, _, _ = model(image)
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
            _, _, _, _, _, pcx_batch, _, _, _ = model(image)
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

    if dataset == 'cifar-10' or dataset == 'cifar-10-eval':
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

    elif dataset == 'cifar-100' or dataset == 'cifar-100-eval':
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
        
    elif dataset == 'cifar-20-eval':
        dataset_class = CIFAR100Coarse
        data_dir = 'data/CIFAR20'
        if normalize:
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
        # test_classes = sorted(list(set(label for _, label in test_set)))
        # test_set = EpisodeDataset(
        #     dataset=test_set,
        #     class_list=test_classes,
        #     n_way=N_WAY_TEST,
        #     k_shot=K_SHOT_TEST,
        #     n_query=N_QUERY_TEST,
        #     n_episodes_per_epoch=N_TEST_EPISODES # Generate enough episodes for evaluation
        # )
        # radnomly split the dataset into 2 parts: 80% for training and 20% for testing

    elif dataset == 'svhn':
        # SVHN has train and test splits
        train_set = dataset_class(root=data_dir, split='train', download=download, transform=transform)
        test_set = dataset_class(root=data_dir, split='test', download=download, transform=transform)
        # radnomly split the dataset into 2 parts: 80% for training and 20% for testing

    elif dataset == 'cifar-10' or dataset == 'cifar-100':
        train_set = dataset_class(root=data_dir, train=True, download=download, transform=transform)
        train_set_aug = SimCLRDatasetWrapper(train_set, crop_size=32)
        test_set = dataset_class(root=data_dir, train=False, download=download, transform=transform)

    elif dataset == 'cifar-10-eval' or dataset == 'cifar-100-eval' or dataset == 'cifar-20-eval':
        train_set = dataset_class(root=data_dir, train=True, download=download, transform=transform)
        # train_set_aug = SimCLRDatasetWrapper(train_set, crop_size=32)
        test_set = dataset_class(root=data_dir, train=False, download=download, transform=transform)


    elif dataset == 'stl-10':
        # STL-10 has train and test splits
        train_set = dataset_class(root=data_dir, split='unlabeled', download=download, transform=transform)
        train_set_aug = SimCLRDatasetWrapper(train_set, crop_size=96)
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

    if dataset == 'cifar-10' or dataset == 'cifar-100' or dataset == 'stl-10':
        train_loader = DataLoader(
            train_set_aug,
            batch_size=batch_size, # This N determines the size of view1_batch and view2_batch
            shuffle=True,
            num_workers=8,
            pin_memory=True, # Recommended for faster GPU transfer
            drop_last=True   # Often used in contrastive learning
        )
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    # if dataset == 'omniglot':
    #     test_loader = DataLoader(
    #         dataset=test_set,
    #         batch_size=None, # EpisodeDataset yields full episodes as batches
    #         num_workers=0 # Often safer for IterableDatasets, adjust if needed
    #     )
    # else:
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False,
                             num_workers=8, pin_memory=True)

    return train_loader, test_loader, train_set, test_set

class CIFAR100Coarse(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
class SimCLRDatasetWrapper(Dataset):
    """
    Wraps a dataset to produce two augmented views for SimCLR.
    """
    def __init__(self, base_dataset, crop_size=32):
        """
        Args:
            base_dataset: The underlying dataset (e.g., CIFAR10 instance).
            transform: The SimCLR augmentation transform to apply twice.
        """
        self.base_dataset = base_dataset
        self.crop_size = crop_size
        self.transform = self.get_augmentation()

    def __len__(self):
        return len(self.base_dataset)
    
    def get_augmentation(self):
        aug = transforms.Compose([
            transforms.RandomResizedCrop(size=[self.crop_size, self.crop_size], scale=(0.2, 1.0), ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.1, 0.1))
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        return aug

    def __getitem__(self, idx):
        # Get the raw image (and possibly label, though often ignored in SimCLR pretraining)
        # Adjust based on what base_dataset returns
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            # Common case: (image, label)
            image = item[0]
            # label = item[1] # We might not need the label for SimCLR pre-training itself
        else:
            # Case: dataset returns only images
            image = item

        # Apply the same augmentation pipeline twice to the *original* image
        # The randomness within the transforms ensures different views
        view1 = self.transform(image)
        view2 = self.transform(image)

        # Return the two views. The DataLoader will handle batching.
        # If you needed labels later (e.g., for linear evaluation), you could return:
        # return view1, view2, label
        return view1, view2


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




############## Eval Metrics ##############

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

def compute_soft_dendrogram_purity_test_only(model, test_dataloader, device, epsilon=1e-9):

    model.eval()
    model.to(device)

    n_classes = 10
    n_nodes = None

    print("Processing test data to get probability distributions (pcx)...")
    all_pcx = []
    all_true_labels = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_dataloader, desc="Evaluating Test Set"):
            if n_classes is None:
                n_classes = int(y_batch.max().item()) + 1
            else:
                n_classes = max(n_classes, int(y_batch.max().item()) + 1)

            x_batch = x_batch.to(device)
            try:
                outputs = model(x_batch)
                if len(outputs) < 6:
                     raise ValueError(f"Model output tuple has length {len(outputs)}, expected at least 6.")
                pcx_batch = outputs[5] 
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print("Check model definition and output structure.")
                return None

            if n_nodes is None:
                n_nodes = pcx_batch.shape[1]
            elif pcx_batch.shape[1] != n_nodes:
                 print(f"Error: Inconsistent number of nodes in model output ({pcx_batch.shape[1]} vs {n_nodes}).")
                 return None

            all_pcx.append(pcx_batch.cpu())
            all_true_labels.append(y_batch.cpu())

    if n_nodes is None or n_classes is None:
        print("Error: Could not determine number of nodes or classes from test data.")
        return None

    try:
        all_pcx = torch.cat(all_pcx).float()
        all_true_labels = torch.cat(all_true_labels)
    except RuntimeError as e:
         print(f"Error concatenating tensors (likely memory issue): {e}")
         print("Try reducing batch size or running on a machine with more memory.")
         return None

    n_test = len(all_true_labels)
    print(f"Processed {n_test} test samples. Found {n_classes} classes and {n_nodes} nodes.")

    all_pcx = all_pcx.to(device)
    all_true_labels = all_true_labels.to(device) 

    print("Calculating node purities based on test set expected counts...")
    test_node_purity = torch.zeros((n_classes, n_nodes), dtype=torch.float32, device=device)

  
    expected_total_count_per_node = torch.sum(all_pcx, dim=0) # Shape: (n_nodes,)

    for k in range(n_classes):
        indices_k = (all_true_labels == k).nonzero(as_tuple=True)[0]
        if len(indices_k) > 0:
            expected_class_count_per_node = torch.sum(all_pcx[indices_k, :], dim=0) # Shape: (n_nodes,)

            denominator = expected_total_count_per_node + epsilon
            test_node_purity[k, :] = expected_class_count_per_node / denominator

    print("Node purities calculated.")

    print("Calculating Soft Dendrogram Purity (iterating over pairs)...")
    total_purity_sum = 0.0
    total_pairs = 0

    for k in tqdm(range(n_classes), desc="Processing Classes"):
        indices_k = (all_true_labels == k).nonzero(as_tuple=True)[0]
        N_k = len(indices_k)

        if N_k < 2:
            continue

        num_pairs_k = N_k * (N_k - 1) / 2
        total_pairs += num_pairs_k
        class_purity_sum = 0.0

        pcx_k = all_pcx[indices_k] # Shape: (Nk, n_nodes)
        purities_k_vector = test_node_purity[k, :] # Shape: (n_nodes,)

        for i_idx in range(N_k):
            P_i = pcx_k[i_idx, :] 
            for j_idx in range(i_idx + 1, N_k):
                P_j = pcx_k[j_idx, :] 

                joint_p = P_i * P_j # Shape: (n_nodes,)

                joint_p_sum = joint_p.sum()
                if joint_p_sum < epsilon:
                    pair_purity = 0.0
                else:
                    weights = joint_p / joint_p_sum
                    pair_purity = torch.dot(weights, purities_k_vector).item() 

                class_purity_sum += pair_purity 

        total_purity_sum += class_purity_sum

    if total_pairs == 0:
        print("Warning: No valid pairs found to calculate purity (test set might be too small or lack classes with >= 2 points).")
        return 0.0

    final_soft_dendrogram_purity = total_purity_sum / total_pairs
    print("Calculation complete.")

    return final_soft_dendrogram_purity


