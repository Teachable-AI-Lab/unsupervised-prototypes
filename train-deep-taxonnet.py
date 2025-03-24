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
# tsne and pca
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import untils
from DeepTaxonNet import CobwebNN, CobwebNNTreeLayer, TestModel
import argparse
import os
import sys
import wandb

MODEL_SAVE_PATH_PREFIX = '/home/zwang910/file_storage/nips-2025/deep-taxon/project-checkin'

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

#######################################################
# define args
parser = argparse.ArgumentParser()
## Training args
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--model_save_path', type=str, default='')
parser.add_argument('--model_save_interval', type=int, default=20)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--linear_probing_lr', type=float, default=5e-2)
parser.add_argument('--linear_probing_epochs', type=int, default=50)
## Model args
parser.add_argument('--n_layers', type=int, default=8)
parser.add_argument('--n_hidden', type=int, default=512)
parser.add_argument('--kl_weight', type=float, default=200.0)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--commitment_weight', type=float, default=1.0)
parser.add_argument('--loss_fn', type=str, default='mse') # or bce # We use MSE
parser.add_argument('--layer_wise_loss', type=bool, default=False)
parser.add_argument('--sampling', type=bool, default=False)
parser.add_argument('--simple_encoder', type=bool, default=False)
## Data args
parser.add_argument('--dataset', type=str, default='cifar-10')
parser.add_argument('--normalize', type=bool, default=False)
## Wandb args
parser.add_argument('--wandb', type=bool, default=False)
parser.add_argument('--wandb_project', type=str, default='deep-taxon')
parser.add_argument('--wandb_run_name', type=str, default='mse-unnormalized')

args = parser.parse_args()
set_seed(args.seed)
device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

#######################################################
# wandb init
if args.wandb:
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    wandb.config.update(args)

print(args.normalize)
print(f"layer_wise_loss: {args.layer_wise_loss}")
#######################################################

# define data loader
print('Loading data...')
train_loader, test_loader, train_data, test_data = untils.get_data_loader(args.dataset, args.batch_size, args.normalize)

# define model
n_layers = args.n_layers
n_hidden = args.n_hidden
image_shape = (3, 32, 32)
image_shape_prod = image_shape[0] * image_shape[1] * image_shape[2]
cobweb = CobwebNN(image_shape=image_shape, n_layers=n_layers, n_hidden=n_hidden,
                  disable_decoder_sigmoid=args.normalize, tau=args.tau, 
                  layer_wise=args.layer_wise_loss, sampling=args.sampling, simple_encoder=args.simple_encoder
                  ).to(device)

# # print model's device
# print(f'Model device: {next(cobweb.parameters()).device}')

# define optimizer
optimizer = optim.AdamW(cobweb.parameters(), lr=args.lr)

print('Start training...')
steps = 0
epochs = args.epochs
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for j, (data, target) in enumerate(train_loader):
        # cobweb.train()
        optimizer.zero_grad()

        data = data.to(device)
        x, means, logvars, x_preds, p_x_nodes, p_node_xs, x_latent, _ = cobweb(data)
        loss = 0
        PTY = 0
        CMT = 0
        REC = 0
        KL = 0
        # break
        H = 0

        ########### layer-wise loss ############
        if args.layer_wise_loss:
            # print(len(means), len(logvars), len(x_preds), len(p_x_nodes), len(p_node_xs))
            for i, (mean, logvar, x_pred, p_node_x) in enumerate(zip(means, logvars, x_preds, p_node_xs)):
                # reconstruction loss
                if args.loss_fn == 'mse':
                    REC += F.mse_loss(x_pred.view(-1, image_shape_prod), data.view(-1, image_shape_prod))
                elif args.loss_fn == 'bce': # can't use this if the data is normalized
                    REC += F.binary_cross_entropy(x_pred.view(-1, image_shape_prod), data.view(-1, image_shape_prod))
                # prototype loss
                # PTY += ((x_latent.detach().unsqueeze(1) - mean.unsqueeze(0)).pow(2).mean(dim=-1) * p_node_x).sum(dim=-1).mean()
                # commitment loss
                # CMT += ((x_latent.unsqueeze(1) - mean.detach().unsqueeze(0)).pow(2).mean(dim=-1) * p_node_x).sum(dim=-1).mean()
                if args.sampling:
                    # maximize the logprob of x_latent given the means and logvars
                    #  dist = -0.5 * torch.sum(logvars_cat + ((x_mu.unsqueeze(1) - means_cat.unsqueeze(0)) ** 2) / torch.exp(logvars_cat), dim=-1)
                    CMT += 0.5 * torch.sum(logvar + ((x_latent.unsqueeze(1) - mean.unsqueeze(0)) ** 2) / torch.exp(logvar), dim=-1).mean()
                else:
                    CMT += ((x_latent.unsqueeze(1) - mean.unsqueeze(0)).pow(2).mean(dim=-1) * p_node_x).sum(dim=-1).mean()
                KL += untils.cross_entropy_regularization(p_node_x, depth=n_layers - i, lambda_=args.kl_weight)
                # print(f"layer {i}: KL: {KL}")

            loss += REC + args.commitment_weight * CMT + KL

        ########### tree-wise loss ############
        else:
            # means: 2**n_layers-1, n_hidden
            # logvars: 2**n_layers-1, n_hidden
            # x_preds: B, 512, 1, 1
            # p_node_xs: B, 2**n_layers-1
            # reconstruction loss
            if args.loss_fn == 'mse':
                REC = F.mse_loss(x_preds.view(-1, image_shape_prod), data.view(-1, image_shape_prod))
            elif args.loss_fn == 'bce': # can't use this if the data is normalized
                REC = F.binary_cross_entropy(x_preds.view(-1, image_shape_prod), data.view(-1, image_shape_prod))
            # prototype loss
            # PTY = ((x_latent.detach().unsqueeze(1) - torch.cat(means, dim=0).unsqueeze(0)).pow(2).mean(dim=-1) * p_node_xs).sum(dim=-1).mean()
            # commitment loss
            # CMT = ((x_latent.unsqueeze(1) - torch.cat(means, dim=0).detach().unsqueeze(0)).pow(2).mean(dim=-1) * p_node_xs).sum(dim=-1).mean()
            if args.sampling:
                # maximize the logprob of x_latent given the means and logvars
                #  dist = -0.5 * torch.sum(logvars_cat + ((x_mu.unsqueeze(1) - means_cat.unsqueeze(0)) ** 2) / torch.exp(logvars_cat), dim=-1)
                CMT += 0.5 * torch.sum(logvars + ((x_latent.unsqueeze(1) - torch.cat(means, dim=0).unsqueeze(0)) ** 2) / torch.exp(logvars), dim=-1).mean()
            else:
                CMT += ((x_latent.unsqueeze(1) - torch.cat(means, dim=0).unsqueeze(0)).pow(2).mean(dim=-1) * p_node_xs).sum(dim=-1).mean()
            KL = untils.cross_entropy_regularization(p_node_xs, depth=n_layers + 1, lambda_=args.kl_weight, entire_tree=(not args.layer_wise_loss))
            # print(f"KL: {KL}")
            loss = REC + args.commitment_weight * CMT + KL
            # print(p_node_xs.shape)
            # print(torch.cat(means, dim=0).shape)

        if args.wandb:
            wandb.log({'loss': loss.item(), 
                       'rec_loss': REC.item(), 
                       'kl_loss': KL.item(), 
                       'pty_loss': PTY,
                       'cmt_loss': CMT.item(),
                       'steps': steps}
                       )

        loss.backward()
        optimizer.step()
        steps += 1
        # break

        # all_losses.append(loss.item())
        # all_rec_losses.append(REC.item())
        # all_kl_losses.append(KL.item())
        # all_pty_losses.append(PTY.item())
        # all_cmt_losses.append(CMT.item())
    # for every epoch
    viz_fig = untils.viz_examplar(cobweb, test_data, n_data=1000, device=device, layer=5, k=10, normalize=args.normalize)
    if args.wandb:
        wandb.log({'viz_examplar': viz_fig, 'epoch': epoch})

    viz_centroids = untils.viz_examplar(cobweb, test_data, n_data=1000, device=device, layer=5, k=10, normalize=args.normalize, do_centroids=True)
    if args.wandb:
        wandb.log({'viz_centroids': viz_centroids, 'epoch': epoch})

    

    # save model every 20 epochs
    if epoch % args.model_save_interval == 0:
        model_save_path = f'{MODEL_SAVE_PATH_PREFIX}/{args.model_save_path}/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # evaluate model via linear probing
        acc = untils.model_forzen_classification(cobweb, train_data, test_data, device=device, lr=args.linear_probing_lr, epochs=args.linear_probing_epochs, batch_size=args.batch_size)
        if args.wandb:
            wandb.log({'linear_probe_acc': acc, 'epoch': epoch})
        
        torch.save(cobweb.state_dict(), f'{model_save_path}/deep_taxon_{epoch}.pt')
        print(f'Model saved at {model_save_path}/deep_taxon_{epoch}.pt')

        
