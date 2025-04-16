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
import utils
from DeepTaxonNet import DeepTaxonNet
import argparse
import os
import sys
import wandb

MODEL_SAVE_PATH_PREFIX = '/nethome/zwang910/file_storage/nips-2025/deep-taxon/project-checkin'

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
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--model_save_path', type=str, default='')
parser.add_argument('--model_save_interval', type=int, default=20)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--linear_probing_lr', type=float, default=1e-2)
parser.add_argument('--linear_probing_epochs', type=int, default=50)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--pretraining_epochs', type=int, default=0)
parser.add_argument('--pretraining_lr', type=float, default=1e-3)
parser.add_argument('--kl1_weight', type=float, default=1.0)
## Model args
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--latent_dim', type=int, default=768)
parser.add_argument('--enc_hidden_dim', type=int, default=64*8*8)
parser.add_argument('--dec_hidden_dim', type=tuple, default=(64,8,8))
parser.add_argument('--encoder_name', type=str, default='resnet18_light')
parser.add_argument('--decoder_name', type=str, default='resnet18_light')
## Data args
parser.add_argument('--dataset', type=str, default='cifar-10')
parser.add_argument('--normalize', type=bool, default=False)
## Wandb args
parser.add_argument('--wandb', type=bool, default=False)
parser.add_argument('--wandb_project', type=str, default='deep-taxon')
parser.add_argument('--wandb_run_name', type=str, default='VaDE-taxon')

args = parser.parse_args()
set_seed(args.seed)
device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

#######################################################
# wandb init
if args.wandb:
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    wandb.config.update(args)

#######################################################

# define data loader
print('Loading data...')
train_loader, test_loader, train_data, test_data = utils.get_data_loader(args.dataset, args.batch_size, args.normalize)

# define model
model = DeepTaxonNet(
    n_layers=args.n_layers,
    input_dim=3*32*32,
    enc_hidden_dim=args.enc_hidden_dim,
    dec_hidden_dim=args.dec_hidden_dim,
    latent_dim=args.latent_dim,
    encoder_name=args.encoder_name,
    decoder_name=args.decoder_name,
    kl1_weight=args.kl1_weight,
).to(device)

# pretraining only on encoder and decoder
if args.pretraining_epochs > 0:
    print('Pretraining encoder and decoder for {} epochs...'.format(args.pretraining_epochs))
    # optimizer_pretrain = optim.AdamW(
    #     list(model.encoder.parameters()) + 
    #     list(model.decoder.parameters()) + 
    #     list(model.fc_mu.parameters()),
    #     lr=args.pretraining_lr
    # )
    # utils.pretrain(model, train_loader, optimizer_pretrain, args.pretraining_epochs, device) 
    optimizer_pretrain = optim.AdamW(model.parameters(), lr=args.pretraining_lr)
    utils.pretrain(model, train_loader, optimizer_pretrain, args.pretraining_epochs, device)

# define optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

print('Start training...')
steps = 0
epochs = args.epochs
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for j, (data, target) in enumerate(train_loader):
        # model.train()
        optimizer.zero_grad()

        data = data.to(device)

        # beta = utils.linear_annealing(epoch, anneal_epochs=100)     
        # model.kl1_weight = beta

        loss, recon_loss, kl1, kl2, _, _, _ = model(data)   


        if args.wandb:
            wandb.log({'loss': loss.item(),
                        'recon_loss': recon_loss.item(),
                        'kl1': -kl1.item(),
                        'kl2': -kl2.item(),
                       'steps': steps})

        loss.backward()
        optimizer.step()
        steps += 1

    # for every epoch
    all_latent, all_labels, pcx, pis, centroid_list, H = utils.get_latent(model, test_loader, device)
    # viz_fig = untils.viz_examplar(model, test_data, n_data=1000, device=device, layer=5, k=10, normalize=args.normalize)
    viz_tsne = utils.plot_tsne(all_latent, all_labels)
    if args.wandb:
        wandb.log({'t-sne': wandb.Image(viz_tsne), 'epoch': epoch})

    viz_pcx = utils.plot_qcx(pcx)
    if args.wandb:
        wandb.log({'viz_pcx': wandb.Image(viz_pcx), 'epoch': epoch})

    viz_pi = utils.plot_pi(pis)
    if args.wandb:
        wandb.log({'viz_pi': wandb.Image(viz_pi), 'epoch': epoch})

    viz_entropy = utils.plot_entropy(H)
    if args.wandb:
        wandb.log({'viz_entropy': wandb.Image(viz_entropy), 'epoch': epoch})

    for layer in range(5):
        viz_centroid = utils.plot_centroids(centroid_list, layer)
        if args.wandb:
            wandb.log({f'viz_centroid_{layer}': wandb.Image(viz_centroid), 'epoch': epoch})
        
        viz_gen = utils.plot_generated_examples(model, layer, 10, device)
        if args.wandb:
            wandb.log({f'viz_gen_{layer}': wandb.Image(viz_gen), 'epoch': epoch})

        viz_example = utils.plot_dataset_examples(model, all_latent, pcx, layer, 10, device)
        if args.wandb:
            wandb.log({f'viz_example_{layer}': wandb.Image(viz_example), 'epoch': epoch})

    # save model every 20 epochs
    if epoch % args.model_save_interval == 0:
        model_save_path = f'{MODEL_SAVE_PATH_PREFIX}/{args.model_save_path}/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # evaluate model via linear probing
        acc = utils.linear_probing(model, args.n_classes, train_loader, test_loader, lr=args.linear_probing_lr, epochs=args.linear_probing_epochs, device=device)
        if args.wandb:
            wandb.log({'linear_probe_acc': acc, 'epoch': epoch})
        
        torch.save(model.state_dict(), f'{model_save_path}/deep_taxon_{epoch}.pt')
        print(f'Model saved at {model_save_path}/deep_taxon_{epoch}.pt')

        
