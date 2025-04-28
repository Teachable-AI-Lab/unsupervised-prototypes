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
from encoder_decoder.resnet_encoder import Encoder, BasicBlockEnc
from encoder_decoder.restnet_decoder import Decoder, BasicBlockDec
from classes.resnet_using_light_basic_block_encoder import LightEncoder, LightBasicBlockEnc
from classes.resnet_using_light_basic_block_decoder import LightDecoder, LightBasicBlockDec
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
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_hidden = 512
        self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2])
        # self.prototype_encoder = prototype_encoder
        self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2], disable_decoder_sigmoid=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = AE().to(device)

# # print model's device
# print(f'Model device: {next(cobweb.parameters()).device}')

# define optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

print('Start training...')
steps = 0
epochs = args.epochs
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    for j, (data, target) in enumerate(train_loader):
        # cobweb.train()
        optimizer.zero_grad()

        data = data.to(device)
        # x, means, logvars, x_preds, p_x_nodes, p_node_xs, x_latent, _ = cobweb(data)
        x = model(data)
        loss = F.mse_loss(x.view(-1, 32*32*3), data.view(-1, 32*32*3))

        if args.wandb:
            wandb.log({'loss': loss.item(), 
                       'steps': steps}
                       )

        loss.backward()
        optimizer.step()
        steps += 1

    # for every epoch
    # viz_fig = untils.viz_examplar(cobweb, test_data, n_data=1000, device=device, layer=5, k=10, normalize=args.normalize)
    # if args.wandb:
    #     wandb.log({'viz_examplar': viz_fig, 'epoch': epoch})

    # viz_centroids = untils.viz_examplar(cobweb, test_data, n_data=1000, device=device, layer=5, k=10, normalize=args.normalize, do_centroids=True)
    # if args.wandb:
    #     wandb.log({'viz_centroids': viz_centroids, 'epoch': epoch})

    

    # save model every 20 epochs
    if epoch % args.model_save_interval == 0:
        # model_save_path = f'{MODEL_SAVE_PATH_PREFIX}/{args.model_save_path}/'
        # if not os.path.exists(model_save_path):
            # os.makedirs(model_save_path)
        # evaluate model via linear probing
        acc = untils.model_forzen_classification(model, train_data, test_data, device=device, lr=args.linear_probing_lr, epochs=args.linear_probing_epochs, batch_size=args.batch_size)
        if args.wandb:
            wandb.log({'linear_probe_acc': acc, 'epoch': epoch})
        
        # torch.save(cobweb.state_dict(), f'{model_save_path}/deep_taxon_{epoch}.pt')
        # print(f'Model saved at {model_save_path}/deep_taxon_{epoch}.pt')

        
