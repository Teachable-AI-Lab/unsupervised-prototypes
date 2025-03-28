# define several helper functions for cobweb-nn experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# umap
# import umap
from IPython.display import clear_output

# define the model output class
class ModelOutput:
    def __init__(self,
                 loss: float=None,
                 centroids: list=None,
                 layer_outputs: list=None,
                 reconstructions: list=None,
                 x: torch.Tensor=None,
                logits: torch.Tensor=None,
                debug_info: dict=None):
        
        self.loss = loss
        self.centroids = centroids
        self.layer_outputs = layer_outputs
        self.reconstructions = reconstructions
        self.x = x
        self.logits = logits
        self.debug_info = debug_info

# define the model
def filter_by_label(mnist_data, labels_to_filter, rename_labels=False):
    filtered_data = []
    for data, label in tqdm(mnist_data):
        if label in labels_to_filter:
            filtered_data.append((data, label))

    if rename_labels:
        new_labels = {label: i for i, label in enumerate(labels_to_filter)}
        filtered_data = [(data, new_labels[label]) for data, label in filtered_data]
    return filtered_data

# def visualize_centroids(centroids, layers=None, do_t_sne=False, do_pca=False):
#     pass

def visualize_decision_boundary(model, val_data, layer=0, n_hidden=784):
    decisions = []
    test_examples = []
    test_targets = []
    test_representations = []

    test_loader = torch.utils.data.DataLoader(val_data, batch_size=512, shuffle=True)

    for i, (d, t) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            outputs = model(d.to('cuda'), t.to('cuda'))
            test_examples.append(d)
            test_targets.append(t)
            # print(rep.shape)
            test_representations.append(outputs.x.detach().cpu())
            # print(outputs[0])
            # print(outputs[0].argmax(dim=-1).tolist())
            # break
            # print(outputs[0].shape)
            # print(outputs[2])
            decisions.extend(outputs.layer_outputs[layer].argmax(dim=-1).tolist())
            if len(decisions) > 1000:
                break

    # plot the data, color by the decision
    # do tsne first on the data
    tsne = TSNE(n_components=2)
    tsne_outputs = tsne.fit_transform(torch.cat(test_representations).view(-1, n_hidden).numpy())
    tsne_labels = torch.cat(test_targets).numpy()
    plt.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=decisions, cmap='tab10')
    plt.colorbar()

    # create a new plot for the decision
    plt.figure()
    plt.scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c=tsne_labels, cmap='tab10')
    plt.colorbar()


# def visualize_filters(model, layers=None):
#     pass

def train_model(model, train_data=None, supervised=False, optimizer=None, device='cuda',
                batch_size=32, epochs=1, hard=False,
                show_loss=False, show_centroids=False,
                show_filters=False, show_decision_boundary=False, verbose=True, early_break=False):
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    losses = []

    model.train()
    for epoch in range(epochs):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as t:
            for x, y in t:
                optimizer.zero_grad()

                if supervised:
                    outputs = model(x.to(device), y.to(device), hard=hard)
                else:
                    outputs = model(x.to(device), hard=hard)

                loss = outputs.loss
                losses.append(loss.item())

                if verbose:
                    print(f"layer outputs: {outputs.layer_outputs}")
                
                loss.backward()
                if early_break:
                    break 
                optimizer.step()

                t.set_postfix(loss=loss.item())


    if show_loss:
        plt.plot(losses)
        plt.show()    

def GumbelSigmoid(logits, tau=1, alpha=1, hard=False, dim=-1):
    def _gumbel():
        gumbel = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbel).sum() or torch.isinf(gumbel).sum():
            gumbel = _gumbel()
        return gumbel
    
    gumbel = _gumbel()
    gumbel = (logits + gumbel * alpha) / tau
    y_soft = F.sigmoid(gumbel)
    if hard:
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def GumbelSoftmax(logits, tau=1, alpha=1, hard=False, dim=-1):
    def _gumbel():
        gumbel = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbel).sum() or torch.isinf(gumbel).sum():
            gumbel = _gumbel()
        return gumbel
    
    gumbel = _gumbel()
    gumbel = (logits + gumbel * alpha) / tau
    y_soft = F.softmax(gumbel, dim=dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def entropy_regularization(left_prob, eps=1e-8, lambda_=1, layer=0):
    entropy = - (left_prob * torch.log(left_prob + eps) + (1 - left_prob) * torch.log(1 - left_prob + eps))
    return entropy * lambda_

def cross_entropy_regularization(path_probs, depth=0, eps=1e-8, lambda_=1, n_layers=3, decay=False, entire_tree=False):
    # path_probs: [batch, 2 * n_clusters]
    assert path_probs.sum(dim=-1).allclose(torch.ones(path_probs.shape[0], device=path_probs.device))
    B = path_probs.shape[0]
    # pp = path_probs.view(B, -1, 2).sum(dim=0) / B
    pp = path_probs.sum(dim=0) / B
    # print(pp)
    a = F.softmax(pp, dim=-1)
    # print(a.shape)
    # print(path_probs.view(B, -1, 2).sum(dim=0), a)
    # equ = 1 / 2 ** (depth)
    equ = -torch.log(torch.tensor(2**(depth), device=path_probs.device))
    if entire_tree:
        # equ = 1 / (2 ** depth - 1)
        equ = -torch.log(torch.tensor(2**(depth) - 1, device=path_probs.device))

    # print(f"nodes: {2 ** depth - 1}, equ: {equ**-1}, nodes: {path_probs.shape[1]}")
    # print(f"equ: {equ}")
    # repeat equ to the shape of a
    equ = torch.tensor([equ] * a.shape[0], device=path_probs.device)
    # print(f"equ: {equ}")
    # reg = (0.5 * torch.log(a) + 0.5 * torch.log(1 - a)).sum()
    # reg = (0.5 * torch.log(a)).sum()
    # kl between a and equ

    # reg = F.kl_div(a.log(), equ, reduction='sum')
    # log kl
    reg = (a * (a.log() - equ)).sum()
    reg = torch.max(reg, torch.tensor(0, dtype=torch.float32, device=path_probs.device))
    if decay:
        lambda_ = lambda_ * torch.log(-torch.tensor(depth - (n_layers + 1), dtype=torch.float32, device=path_probs.device))

    # print(f"Cross entropy regularization at depth {depth}: {reg}")
    # lambda_ = lambda_ * (2 ** (-depth))
    return reg * lambda_

import torch

def layer_kld(mean, logvar, reduction='sum'):
    """
    Computes the KL divergence between each left/right child pair in a tree layer.

    The distributions are assumed to be Gaussian:
      - Left child: N(mean_left, var_left) with var_left = exp(logvar_left)
      - Right child: N(mean_right, var_right) with var_right = exp(logvar_right)

    The KL divergence from left to right is computed as:
        0.5 * [ log(var_right/var_left) + (var_left + (mean_left - mean_right)**2)/var_right - 1 ]

    Parameters:
        mean (torch.Tensor): Tensor of shape (n_pairs, 2, 1), where each pair corresponds to
                             the means of the left and right children.
        logvar (torch.Tensor): Tensor of shape (n_pairs, 2, 1), corresponding log-variances.
        reduction (str): Specifies the reduction over all pairs: 'sum' or 'mean'.

    Returns:
        kl (torch.Tensor): Reduced KL divergence across all pairs (scalar).
        kl_each (torch.Tensor): KL divergence for each pair, shape (n_pairs, 1).
    """
    # Separate left and right children parameters
    mean_left = mean[:, 0, :]   # shape: (n_pairs, 1)
    mean_right = mean[:, 1, :]  # shape: (n_pairs, 1)
    logvar_left = logvar[:, 0, :]
    logvar_right = logvar[:, 1, :]

    # Compute variance from log variance
    var_left = torch.exp(logvar_left)
    var_right = torch.exp(logvar_right)

    # Compute the KL divergence for each pair:
    # KL(N_left || N_right) = 0.5 * ( log(var_right/var_left) + (var_left + (mean_left - mean_right)^2) / var_right - 1 )
    kl_each = 0.5 * (logvar_right - logvar_left + (var_left + (mean_left - mean_right) ** 2) / var_right - 1)

    # Apply the specified reduction across all pairs
    if reduction == 'sum':
        kl = kl_each.sum()
    elif reduction == 'mean':
        kl = kl_each.mean()
    else:
        raise ValueError("Reduction must be either 'sum' or 'mean'.")

    return kl, kl_each

def test_model(model, test_data, device='cuda', batch_size=32, hard=False, leaf_only=False):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    labels = []
    pred = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x.to(device), y.to(device), hard=hard, leaf_only=leaf_only)
            labels.extend(y.tolist())
            pred.extend(outputs.logits.argmax(dim=-1).tolist())

        # calculate the accuracy
        correct = 0
        for l, p in zip(labels, pred):
            if l == p:
                correct += 1
        accuracy = correct / len(labels)
        # print(f"Accuracy: {accuracy}")
        return accuracy


def get_tree_data(model=None, test_data=None, filename=None, image_data=None, image_shape=None, 
                  dist_images=None, dist_shape=None):
    x_splits = []
    layer_logits = []
    layer_z = []

    model.eval()
    with torch.no_grad():
        x, y = test_data
        outputs = model(x.to('cuda'))
        x_splits.append(torch.cat([x_split.detach().cpu() for x_split in outputs.debug_info['x_splits']], dim=1))
        layer_logits.append(torch.cat([logits.detach().cpu() for logits in outputs.debug_info['layer_logits']], dim=1))
        layer_z.append(torch.cat([z.detach().cpu() for z in outputs.debug_info['layer_z']], dim=1))

    x_splits = torch.cat(x_splits, dim=0) # shape of x_splits: [batch, 2 * n_clusters, n_hidden]
    layer_logits = torch.cat(layer_logits, dim=0) # shape of layer_logits: [batch, 2 * n_clusters, n_classes]
    layer_z = torch.cat(layer_z, dim=0) # shape of layer_z: [batch, 2 * n_clusters, n_hidden]

    # shape of x_splits: [batch, 2 * n_clusters, n_hidden]
    # shape of layer_logits: [batch, 2 * n_clusters, n_classes]
    # both x_splits and layer_logits are organized as follows (the n_clusters dimension):
    # [left1, left2, ..., leftN, right1, right2, ..., rightN]
    # want to reorganize them in complete binary tree order as follows:
    # [left1, right1, left2, right2, ..., leftN, rightN]



    
### Visualization

import base64
import matplotlib.pyplot as plt
from io import BytesIO
import json

def tensor_to_base64(tensor, shape, cmap="gray", normalize=False):
    # print(tensor.shape)

    # tensor has shape 1, channel, height, width
    # array = tensor.view(shape)
    # display image with rgb
    # if normalize:
    array = tensor.view(shape).permute(1, 2, 0).cpu().numpy() 
    # denormalize the image with mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    array = array * [0.247, 0.243, 0.261] + [0.4914, 0.4822, 0.4465]
    # normalize from 0-1
    # array = (array - array.min()) / (array.max() - array.min())
    # print(array)
    # return
    plt.imshow(array)
    # plt.show()

        # plt.imshow(array, cmap=cmap, aspect="auto")
    # else:
        # plt.imshow(array, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def viz_prototypes(model, shape=(3, 32, 32), conved_shape=(64, 5, 5), normalize=False):
    nodes = []
    leaves = model.leaves.detach()
    with torch.no_grad():
        # rec_leaves = model.decoder(leaves).detach().cpu()
        rec_leaves = model.decoder(model.decoder_fc(leaves).view(-1, conved_shape[0], conved_shape[1], conved_shape[2])).detach()
        # print(leaves.shape)
    nodes.append(rec_leaves)

    for layer in model.layers:
        alpha = F.sigmoid(layer.cluster_weight.detach())
        alpha = torch.cat((alpha, 1 - alpha), dim=1).unsqueeze(-1)
        # print(leaves.shape)
        # print(leaves.shape)

        leaves = leaves.view(-1, 2, leaves.shape[-1])
        # print("leaf", leaves)

        parent_mean = (alpha * leaves).sum(dim=1) # shape: n_clusters, n_hidden
        # print(alpha)
        # print("parent",parent_mean)
        with torch.no_grad():
            # rec_leaves = model.decoder(parent_mean).detach().cpu()
            rec_leaves = model.decoder(model.decoder_fc(parent_mean).view(-1, conved_shape[0], conved_shape[1], conved_shape[2])).detach()
        # print(rec_leaves.shape)
        nodes.append(rec_leaves)

        leaves = parent_mean

    # reverse the nodes
    nodes = nodes[::-1] # 1node, 2node, 4node, 8node, ..., 2^Nnode
    # print(nodes[0].shape)
    viz_data(nodes, shape=shape, normalize=normalize)

def viz_data(nodes, shape=(28, 28), normalize=True):
    child_idx = 1

    root = {
        "node_id": "0",
        "image": tensor_to_base64(nodes[0][0], shape, cmap="inferno", normalize=normalize),
        # "dist_image": tensor_to_base64(torch.zeros(dist_shape), dist_shape, cmap="viridis"),
        "children": []
    }

    parents = [root]
    for data_list in nodes[1:]:
        print(f"Number of children: {len(data_list)}")
        children_count = 0
        parent = None
        new_parents = []
        for data in data_list:
            if children_count % 2 == 0:
                parent = parents.pop(0)
            # print(data.shape)
            parent["children"].append({
                "node_id": str(child_idx),
                "image": tensor_to_base64(data, shape, cmap="inferno", normalize=True),
                # "dist_image": tensor_to_base64(dist, dist_shape, cmap="viridis", normalize=True),
                "children": []
            })
            child_idx += 1
            children_count += 1
            new_parents.append(parent["children"][-1])
        parents = new_parents

    with open('tree_data.json', "w") as f:
        json.dump(root, f, indent=2)

def viz_batch_reconstruction(model, test_data, batch_size=32, subplot_shape=(4, 8), device='cuda'):

    '''
    DON'T USE FOR NOW
    '''

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    model.eval()
    with torch.no_grad():
        for input_x, _ in test_loader:
            batch_x = model(input_x.to(device), _).reconstructions.detach().cpu()
            break
        # batch_x = input_x

    fig, axes = plt.subplots(*subplot_shape, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(batch_x[i].reshape(28, 28), cmap="inferno")
        ax.axis("off")
    plt.show()

def viz_sampled_x(model, test_data, device='cuda', shape=(28, 28)):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    model.eval()
    with torch.no_grad():
        for input_x, _ in test_loader:
            x, means, logvars, sampled, p_x_nodes = model(input_x.to(device))
            break

    sampled = [s.detach().cpu() for s in sampled][::-1]
    viz_data(sampled, shape=shape, normalize=True)


def viz_clusters(model, test_data, device='cuda', n_data=1000):
    # sub plot of 1x3 for t-sne. PCA
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes = axes.flatten()

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=n_data, shuffle=True)

    # print(len(test_loader))
    batch_size = n_data

    all_representations = []
    all_labels = []

    # leaves = model.leaves.detach()
    # print(leaves.shape)
    prototypes1 = None
    prototypes2 = None
    prototypes3 = None
    # get representations
    with torch.no_grad():
        for input_x, input_y in test_loader:

            x, means, logvars, x_preds, p_x_nodes, p_node_xs, x_latent, x_samples = model(input_x.to(device))
            # print(p_node_xs[0].shape)
            # print(torch.argmax(p_node_xs[0], dim=-1))
            # break

            # input_x = input_x.expand(-1, 3, -1, -1)
        # pad to 32x32
            # input_x = F.pad(input_x, (2, 2, 2, 2), value=0)
            # conved = model.encoder(input_x.to(device)).view(-1, 512).detach().cpu()
            # latent = model.pre_quantization_conv(conved).view(-1, 512).detach().cpu()
            # print(x_samples[0].shape)
            latent = x_samples.detach().cpu()
            # print(conved.shape)
            # latent = model.encoder_fc(conved.view(batch_size, -1))
            # latent = model.encoder_bn(latent).detach().cpu()
            # latent = F.tanh(latent)
            print(latent.min(), latent.max())
            print(model.leaves.min(), model.leaves.max())
            # print(f"alpha: {F.sigmoid(model.layers[0].cluster_weight)}")
            # print(model.prototype_encoder(model.prototype_noise).min(), model.prototype_encoder(model.prototype_noise).max())
            # print(latent.shape)
            all_representations.append(latent)
            all_labels.append(input_y)
            prototypes1 = means[-1].detach().cpu()
            prototypes2 = means[-2].detach().cpu()
            prototypes3 = means[-3].detach().cpu()
            # print(prototypes.shape)
            all_representations.append(prototypes1)
            all_representations.append(prototypes2)
            all_representations.append(prototypes3)

            all_labels.append(torch.tensor([10] * prototypes1.shape[0]))
            all_labels.append(torch.tensor([11] * prototypes2.shape[0]))
            all_labels.append(torch.tensor([12] * prototypes3.shape[0]))
            break

    all_representations = torch.cat(all_representations, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # t-sne
    tsne = TSNE(n_components=2)
    tsne_outputs = tsne.fit_transform(all_representations)
    print(tsne_outputs.shape)
    axes[0].scatter(tsne_outputs[0:1000, 0], tsne_outputs[0:n_data, 1], c=all_labels[0:n_data], cmap='tab10', s=5)

    axes[0].scatter(tsne_outputs[n_data:n_data+1, 0], tsne_outputs[n_data:n_data+1, 1], c='black', s=20)  
    axes[0].scatter(tsne_outputs[n_data+1:n_data+3, 0], tsne_outputs[n_data+1:n_data+3, 1], c='red', s=20)
    axes[0].scatter(tsne_outputs[n_data+3:n_data+7, 0], tsne_outputs[n_data+3:n_data+7, 1], c='blue', s=20)

    
    axes[0].set_title("t-SNE")
    # colorbar
    # axes[0].colorbar()

    # do tsne on the prototypes
    # prototypes = model.leaves.detach().cpu()
    # tsne_outputs = tsne.fit_transform(prototypes)
    # axes[0].scatter(tsne_outputs[:, 0], tsne_outputs[:, 1], c='black', s=3)
    # axes[0].set_title("t-SNE")


    # PCA
    pca = PCA(n_components=2)
    pca_outputs = pca.fit_transform(all_representations)
    axes[1].scatter(pca_outputs[:, 0], pca_outputs[:, 1], c=all_labels, cmap='tab10')
    axes[1].set_title("PCA")
    # colorbar
    # axes[1].colorbar()

    plt.show()


def viz_examplar(model, test_data, n_data=1000, device='cuda', layer=3, k=10, normalize=False, do_centroids=False):
    # get image representations
    recons = None
    image_labels = []

    test_loader = DataLoader(test_data, batch_size=n_data, shuffle=True)
    if do_centroids:
        k = 1

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            # print(data.shape)
            x, means, logvars, x_preds, p_x_nodes, p_node_xs, x_latent, _ = model(data)
            # x_latent: shape (batch_size, n_hidden)
            clusters = means[-layer].detach().cpu() # shape (n_clusters, n_hidden)
            x_latent = x_latent.detach().cpu() # shape (batch_size, n_hidden)
            # print(clusters.shape)
            # get the k nearest neighbors for each cluster
            # get the distance between each x_latent and each cluster
            distances = torch.norm(x_latent.unsqueeze(1) - clusters.unsqueeze(0), p=2, dim=-1).cpu() # shape (batch_size, n_clusters)
            # get the k nearest neighbors
            # print(distances.shape)
            # print(x_latent.shape) # (batch_size, n_hidden)
            # print(distances)
            # print(distances.T) # (n_clusters, batch_size)
            _, topk = distances.T.topk(k, dim=-1, largest=False)
            # print(topk.shape) # (n_clusters, k)
            # print(topk)
            # for each cluster, index the top k nearest neighbors from x_latent, return a n_clusters x k x n_hidden tensor
            if not do_centroids:
                examplars = torch.stack([x_latent[topk[i]] for i in range(topk.shape[0])]).to(device)
                # print(examplars.shape)
                examplars = examplars.view(-1, model.n_hidden) # shape (n_clusters * k, n_hidden)

            if do_centroids:
                examplars = clusters.view(-1, model.n_hidden).to(device) # shape (n_clusters, n_hidden)

            # pass the examplars to the decoder
            x_pred = model.decoder(examplars.view(-1, 512, 1, 1)) # shape (n_clusters * k, 3, 32, 32)
            x_pred = x_pred.view(topk.shape[0], topk.shape[1], 3, 32, 32).detach().cpu() # shape (n_clusters, k, 3, 32, 32)
            # print(x_pred.shape)

            # image_labels.append(target)
            break

    # plot the examplars
    fig, axes = plt.subplots(topk.shape[1], topk.shape[0], figsize=(8, 8),
                            gridspec_kw={'wspace': 0, 'hspace': 0})
    for i in range(topk.shape[0]):
        if topk.shape[1] == 1:
            if normalize:
                img = x_pred[i, 0].permute(1, 2, 0).numpy()
                img = img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
                axes[i].imshow(img)
            else:
                axes[i].imshow(x_pred[i, 0].permute(1, 2, 0).numpy())
            axes[i].axis('off')
            continue
        for j in range(topk.shape[1]):
            if normalize:
                img = x_pred[i, j].permute(1, 2, 0).numpy()
                img = img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
                axes[j, i].imshow(img)
            else:
                axes[j, i].imshow(x_pred[i, j].permute(1, 2, 0).numpy())
            axes[j, i].axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    # plt.show()
    return fig




def model_forzen_classification(model, train_data, test_data, device='cuda', lr=5e-2, epochs=30, batch_size=128):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    labels = []
    pred = []
    model.eval()

    # print model's device
    # print(f'Model device: {next(model.parameters()).device}')


    classifier = nn.Linear(model.n_hidden, 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # print(model.device)
    # print(classifier.device)


    # with torch.no_grad():
    #     for x, y in test_loader:
    #         reper = model.model.encoder_fc(model.encoder(x).view(-1, model.CNN_output_dim))

    # reper = reper.detach()
    all_losses = []
    # train the classifier
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()

            # print(x.shape)
            x = x.to(device)
            # with torch.no_grad():
                # x = x.expand(-1, 3, -1, -1)
                # x, means, logvars, x_preds, p_x_nodes, p_node_xs, x_latent, x_samples = model(x.to(device))
            # reper = x_samples[0].detach().cpu()

            #with torch.no_grad():
                # x = x.expand(-1, 3, -1, -1)
        # pad to 32x32
                # x = F.pad(x, (2, 2, 2, 2), value=0)
                # reper = model.encoder_fc(model.encoder(x).view(-1, model.CNN_output_dim)).detach().cpu()
            with torch.no_grad():
                reper = model.encoder(x).view(-1, 512)#.detach().cpu()
                # reper = model.encoder_fc(reper.view(reper.shape[0], -1)).detach().cpu() # shape (batch_size, n_hidden)
                # reper = model.pre_quantization_conv(reper).view(-1, model.n_hidden).detach()
                # reper = model.encoder_fc(reper.view(-1, 512))
                # reper = model.encoder_bn(reper).detach().cpu()

            logits = classifier(reper.to(device))
            # check if the parameters are updated
            # print(classifier.weight)
            loss = F.cross_entropy(logits, y.to(device))
            loss.backward()
            all_losses.append(loss.item())
            optimizer.step()

    # plt.plot(all_losses)

    # test the classifier
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.no_grad():
                # x = x.expand(-1, 3, -1, -1)
        # pad to 32x32
                # x = F.pad(x, (2, 2, 2, 2), value=0)
                # reper = model.encoder_fc(model.encoder(x).view(-1, model.CNN_output_dim)).detach().cpu()
                reper = model.encoder(x).view(-1, 512)#.detach().cpu()
                # reper = model.pre_quantization_conv(reper).view(-1, model.n_hidden).detach()
                # reper = model.encoder_fc(reper.view(reper.shape[0], -1)).detach().cpu() # shape (batch_size, n_hidden)

            logits = classifier(reper.to(device))
            labels.extend(y.tolist())
            pred.extend(logits.argmax(dim=-1).tolist())

        # calculate the accuracy
        correct = 0
        for l, p in zip(labels, pred):
            if l == p:
                correct += 1
        accuracy = correct / len(labels)
        print(f"Accuracy: {accuracy}")
    model.train()
    return accuracy



##################################################
# data loader

def get_data_loader(dataset, batch_size, normalize):
    if dataset == 'cifar-10':
        # load CIFAR10
        download = True
        dataset_class = datasets.CIFAR10
        if normalize:
            cifar10_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                #  std=[0.247, 0.243, 0.261])
            ])

        cifar10_train = dataset_class('data/CIFAR10', train=True, download=download, transform=cifar10_transform)
        cifar10_test = dataset_class('data/CIFAR10', train=False, download=download, transform=cifar10_transform)

        cifar10_train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        cifar10_test_loader = DataLoader(cifar10_test, batch_size=batch_size*2, shuffle=True, num_workers=4, pin_memory=True)

        return cifar10_train_loader, cifar10_test_loader, cifar10_train, cifar10_test