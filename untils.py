# define several helper functions for cobweb-nn experiments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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

def entropy_regularization(left_prob, eps=1e-8, lambda_=1, layer=0):
    entropy = - (left_prob * torch.log(left_prob + eps) + (1 - left_prob) * torch.log(1 - left_prob + eps))
    return entropy * lambda_

def cross_entropy_regularization(path_probs, depth=0, eps=1e-8, lambda_=1, n_layers=3, decay=False):
    # path_probs: [batch, 2 * n_clusters]
    assert path_probs.sum(dim=-1).allclose(torch.ones(path_probs.shape[0], device=path_probs.device))
    B = path_probs.shape[0]
    # pp = path_probs.view(B, -1, 2).sum(dim=0) / B
    pp = path_probs.sum(dim=0) / B
    # print(pp)
    a = F.softmax(pp, dim=-1)
    # print(path_probs.view(B, -1, 2).sum(dim=0), a)
    equ = 1 / 2 ** (depth + 1)
    # repeat equ to the shape of a
    equ = torch.tensor([equ] * a.shape[0], device=path_probs.device)
    # print(f"equ: {equ}")
    # reg = (0.5 * torch.log(a) + 0.5 * torch.log(1 - a)).sum()
    # reg = (0.5 * torch.log(a)).sum()
    # kl between a and equ

    reg = F.kl_div(a.log(), equ, reduction='sum')
    if decay:
        lambda_ = lambda_ * torch.log(-torch.tensor(depth - (n_layers + 1), dtype=torch.float32, device=path_probs.device))

    # print(f"Cross entropy regularization at depth {depth}: {reg}")
    # lambda_ = lambda_ * (2 ** (-depth))
    return reg * lambda_


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
    array = tensor.numpy().reshape(shape)
    if normalize:
        plt.imshow(array, cmap=cmap, aspect="auto")
    else:
        plt.imshow(array, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    plt.axis("off")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def viz_prototypes(model, shape=(28, 28), normalize=True):
    nodes = []
    leaves = model.leaves.detach().cpu()

    nodes.append(leaves)

    for layer in model.layers:
        alpha = F.sigmoid(layer.cluster_weight.detach().cpu())
        alpha = torch.cat((alpha, 1 - alpha), dim=1).unsqueeze(-1)

        leaves = leaves.view(-1, 2, leaves.shape[-1])

        parent_mean = (alpha * leaves).sum(dim=1) # shape: n_clusters, n_hidden

        nodes.append(parent_mean)

        leaves = parent_mean

    # reverse the nodes
    nodes = nodes[::-1] # 1node, 2node, 4node, 8node, ..., 2^Nnode
    # print(nodes[0].shape)
    viz_data(nodes, shape=shape, normalize=normalize)

def viz_data(nodes, shape=(28, 28), normalize=True):
    child_idx = 1

    root = {
        "node_id": "0",
        "image": tensor_to_base64(nodes[0], shape, cmap="inferno", normalize=normalize),
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