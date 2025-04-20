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
from math import log2  # Import log2 directly to avoid shadowing issues


# ------------------------------
# KL Annealing Scheduler (Linear)
# ------------------------------
def linear_annealing(epoch, anneal_epochs=50):
    # Increase beta linearly until it reaches 1.0
    return min(1.0, (epoch+1) / anneal_epochs)

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

def label_annotation(model, suppoer_set, n_classes, device):
    '''
    support_set: N, input_dim, The training data

    return: a n_classs x n_nodes matrix. Each column stores the class distribution of the corresponding cluster.
    '''
    model.eval()
    pcx = []
    labels = []

    support_loader = DataLoader(suppoer_set, batch_size=512, shuffle=False)

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

    

    # normlize the annotation on each column
    annotation = annotation / torch.sum(annotation, dim=0, keepdim=True)

    return annotation

def basic_node_evaluation(model, annotation, query_set, device):
    # annotation: n_classes x n_nodes
    model.eval()
    pcx = []
    labels = []

    query_loader = DataLoader(query_set, batch_size=512, shuffle=False)

    with torch.no_grad():
        for i, (image, label) in enumerate(query_loader):
            image = image.to(device)
            _, _, _, _, _, pcx_batch, _, _ = model(image)
            pcx.append(pcx_batch.detach().cpu())
            labels.append(label)

    pcx = torch.cat(pcx, dim=0) # shape: (N, n_nodes)
    labels = torch.cat(labels, dim=0) # shape: (N, )

    # find argmax of pcx
    # pred = pcx.argmax(dim=1) # shape: (N, ) # return index? Answer: yes
    # print(f"pred shape: {pred[:10]}")
    # index the columns of annotation 
    # pred = annotation[:, pred] # shape: (n_classes, N)
    # pred = pred.T # shape: (N, n_classes)

    # weight the columns of annotation by pcx
    pred = pcx @ annotation.T # shape: (N, n_classes)
    print(f"pred shape: {pred.shape}")

    pred = pred.argmax(dim=1) # shape: (N, ) # return index? Answer: yes

    # compute accuracy
    correct = (pred == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    # print(f"Accuracy: {acc:.4f}")
    return acc

    # print(pred[0])
    


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
            all_latent.append(latent)
            all_labels.append(target)

            _, recon_loss, kl1, kl2, H, pcx, pi, _ = model(data)
            all_pcx.append(pcx)
            pis = pi.detach().cpu().numpy()
            # pis.append(pi)
            if i == 1:
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


def get_data_loader(dataset, batch_size, normalize):
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
    elif dataset == 'stl10':
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
        test_set = dataset_class(root=data_dir, download=download, transform=transform)

    else:
    # Create the training and testing datasets
        train_set = dataset_class(root=data_dir, train=True, download=download, transform=transform)
        test_set = dataset_class(root=data_dir, train=False, download=download, transform=transform)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size * 8, shuffle=False,
                             num_workers=4, pin_memory=True)

    return train_loader, test_loader, train_set, test_set