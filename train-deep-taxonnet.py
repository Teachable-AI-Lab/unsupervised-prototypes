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
import json

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
# parser = argparse.ArgumentParser()
# ## Training args
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--lr', type=float, default=5e-4)
# parser.add_argument('--model_save_path', type=str, default='')
# parser.add_argument('--model_save_interval', type=int, default=20)
# parser.add_argument('--device_id', type=int, default=0)
# parser.add_argument('--lr_scheduler', type=str, default='none') # none, step, cosine, linear-up, linear-down
# parser.add_argument('--linear_probing_lr', type=float, default=1e-2)
# parser.add_argument('--linear_probing_epochs', type=int, default=50)
# parser.add_argument('--n_classes', type=int, default=10)
# parser.add_argument('--pretraining_epochs', type=int, default=0)
# parser.add_argument('--pretraining_lr', type=float, default=1e-3)
# parser.add_argument('--kl1_weight', type=float, default=1.0)
# parser.add_argument('--recon_weight', type=float, default=1.0)
# parser.add_argument('--vade_baseline', type=bool, default=False)
# ## Model args
# parser.add_argument('--n_layers', type=int, default=4)
# parser.add_argument('--latent_dim', type=int, default=10)
# parser.add_argument('--input_dim', type=int, default=1*28*28)
# parser.add_argument('--enc_hidden_dim', type=int, default=128*1*1)
# parser.add_argument('--dec_hidden_dim', type=tuple, default=(128,1,1))
# parser.add_argument('--encoder_name', type=str, default='omniglot')
# parser.add_argument('--decoder_name', type=str, default='omniglot')
# parser.add_argument('--dkl_margin', type=float, default=0.1)
# parser.add_argument('--dkl_weight_lambda', type=float, default=0.1)
# parser.add_argument('--convex_weight_lambda', type=float, default=0.1)
# ## Data args
# parser.add_argument('--dataset', type=str, default='fashion-mnist')
# parser.add_argument('--normalize', type=bool, default=False)
# ## Wandb args
# parser.add_argument('--wandb', type=bool, default=False)
# parser.add_argument('--wandb_project', type=str, default='deep-taxon')
# parser.add_argument('--wandb_run_name', type=str, default='VaDE-taxon')

# args = parser.parse_args()

parser = argparse.ArgumentParser(
    description="Train Deep TaxonNet using configuration from a JSON file."
)
parser.add_argument(
    '--config',
    type=str,
    required=True,
    help='Path to the JSON configuration file.'
)

# Parse the command line to get the config file path
initial_args = parser.parse_args()

# Load the actual configuration from the specified JSON file
try:
    args = utils.load_config_from_json(initial_args.config)
except (FileNotFoundError, ValueError) as e:
    parser.error(str(e)) # Display error through argparse system

# --- Configuration is loaded into 'args' ---
print("Configuration loaded successfully:")
print("-" * 30)
# Print all loaded args and their types
for key, value in sorted(vars(args).items()): # Sort for consistent output
    print(f"  {key}: {value} (Type: {type(value).__name__})")
print("-" * 30)




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
    input_dim=args.input_dim,
    enc_hidden_dim=args.enc_hidden_dim,
    dec_hidden_dim=args.dec_hidden_dim,
    latent_dim=args.latent_dim,
    encoder_name=args.encoder_name,
    decoder_name=args.decoder_name,
    kl1_weight=args.kl1_weight,
    recon_weight=args.recon_weight,
    dkl_margin=args.dkl_margin,
    dkl_weight_lambda=args.dkl_weight_lambda,
    convex_weight_lambda=args.convex_weight_lambda,
    vade_baseline=args.vade_baseline,
    pretrained_encoder=args.pretrained_encoder,
    logvar_init_range=args.logvar_init_range,
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
scheduler = None
if args.lr_scheduler == 'step':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
elif args.lr_scheduler == 'linear-up':
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs)
elif args.lr_scheduler == 'linear-down':
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.epochs)
elif args.lr_scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-5)
    

print('Start training...')
steps = 0
epochs = args.epochs
for epoch in range(epochs):
    print(f'Epoch {epoch}')
    model.train()
    for j, (data, target) in enumerate(train_loader):
        # model.train()
        optimizer.zero_grad()

        data = data.to(device)

        # beta = utils.linear_annealing(steps, anneal_epochs=40000)     
        # model.kl1_weight = beta

        # if epoch < 10:
        #     model.noise_strength = 0.0
        # elif epoch < 30:
        #     model.noise_strength = 0.1
        # elif epoch < 50:
        #     model.noise_strength = 0.2

        
        # recon_weight, kl1_weight = utils.get_loss_weights(steps, 0, 5, 'dkl')
        # model.kl1_weight = kl1_weight
        # model.recon_weight = recon_weight

        ## psuedo code
        # x_aug_1 = utils.data_augmentation(data)
        # x_aug_2 = utils.data_augmentation(data)
        # x_aug = torch.cat((x_aug_1, x_aug_2), dim=0) # shape (2*batch_size, 1, 28, 28)
        # do SimCLR

        loss, recon_loss, kl1, kl2, _, _, _, _ = model(data)  

        # z: shape (2*batch_size, latent_dim)
        # compute contrastive loss
        # sim_score = torch.matmul(z_contrastive, z_contrastive.T) / 0.5
        # compute NT-Xent loss


         




        if args.wandb:
            wandb.log({'loss': loss.item(),
                        'recon_loss': recon_loss.item(),
                        'kl1': -kl1.item(),
                        'kl2': -kl2.item(),
                        # 'beta': model.logvar_x.item(),
                       'steps': steps})

        loss.backward()
        optimizer.step()
        steps += 1

    # for every epoch
    if scheduler is not None:
        scheduler.step()
    if not args.vade_baseline:
        pass
        # if args.dataset == 'omniglot':
        #     all_latent, all_labels, pcx, pis, centroid_list, H = utils.get_latent(model, train_loader, device)
        # else:
        #     all_latent, all_labels, pcx, pis, centroid_list, H = utils.get_latent(model, test_loader, device)
        # # viz_fig = untils.viz_examplar(model, test_data, n_data=1000, device=device, layer=5, k=10, normalize=args.normalize)
        # viz_tsne = utils.plot_tsne(all_latent, all_labels)
        # if args.wandb:
        #     wandb.log({'t-sne': wandb.Image(viz_tsne), 'epoch': epoch})

        # viz_pcx = utils.plot_qcx(pcx)
        # if args.wandb:
        #     wandb.log({'viz_pcx': wandb.Image(viz_pcx), 'epoch': epoch})

        # # viz_pi = utils.plot_pi(pis)
        # # if args.wandb:
        # #     wandb.log({'viz_pi': wandb.Image(viz_pi), 'epoch': epoch})

        # viz_entropy = utils.plot_entropy(H)
        # if args.wandb:
        #     wandb.log({'viz_entropy': wandb.Image(viz_entropy), 'epoch': epoch})

        # for layer in range(5):
        #     viz_centroid = utils.plot_centroids(centroid_list, layer)
        #     if args.wandb:
        #         wandb.log({f'viz_centroid_{layer}': wandb.Image(viz_centroid), 'epoch': epoch})
            
        #     viz_gen = utils.plot_generated_examples(model, layer, 10, device)
        #     if args.wandb:
        #         wandb.log({f'viz_gen_{layer}': wandb.Image(viz_gen), 'epoch': epoch})

        #     viz_example = utils.plot_dataset_examples(model, all_latent, pcx, layer, 10, device)
        #     if args.wandb:
        #         wandb.log({f'viz_example_{layer}': wandb.Image(viz_example), 'epoch': epoch})

    # save model every 20 epochs
    if epoch % args.model_save_interval == 0:
        if not args.vade_baseline:
            model_save_path = f'{MODEL_SAVE_PATH_PREFIX}/{args.model_save_path}/'
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), f'{model_save_path}/deep_taxon_{epoch}.pt')
            print(f'Model saved at {model_save_path}/deep_taxon_{epoch}.pt')
        # evaluate model via linear probing
        # acc = utils.linear_probing(model, args.n_classes, train_loader, test_loader, lr=args.linear_probing_lr, epochs=args.linear_probing_epochs, device=device)
        # if args.wandb:
            # wandb.log({'linear_probe_acc': acc, 'epoch': epoch})
        if args.dataset == 'omniglot': # few shot learning dataste
            # 5-way 1-shot classification
            _, fs_query_loader, _, _ = utils.get_data_loader('omniglot', 256, False,
                                                            N_WAY_TEST=5,
                                                            K_SHOT_TEST=1,
                                                            N_QUERY_TEST=15,
                                                            N_TEST_EPISODES=1000)
            acc = utils.testing_few_shot(model, fs_query_loader, device)
            if args.wandb:
                wandb.log({'5-way 1-shot acc': acc, 'epoch': epoch})
            
            # 5-way 5-shot classification
            _, fs_query_loader, _, _ = utils.get_data_loader('omniglot', 256, False,
                                                            N_WAY_TEST=5,
                                                            K_SHOT_TEST=5,
                                                            N_QUERY_TEST=15,
                                                            N_TEST_EPISODES=1000)
            acc = utils.testing_few_shot(model, fs_query_loader, device)
            if args.wandb:
                wandb.log({'5-way 5-shot acc': acc, 'epoch': epoch})
            
            # 20-way 1-shot classification
            _, fs_query_loader, _, _ = utils.get_data_loader('omniglot', 256, False,
                                                            N_WAY_TEST=20,
                                                            K_SHOT_TEST=1,
                                                            N_QUERY_TEST=15,
                                                            N_TEST_EPISODES=1000)
            acc = utils.testing_few_shot(model, fs_query_loader, device)
            if args.wandb:
                wandb.log({'20-way 1-shot acc': acc, 'epoch': epoch})

            # 20-way 5-shot classification
            _, fs_query_loader, _, _ = utils.get_data_loader('omniglot', 256, False,
                                                            N_WAY_TEST=20,
                                                            K_SHOT_TEST=5,
                                                            N_QUERY_TEST=15,
                                                            N_TEST_EPISODES=1000)
            acc = utils.testing_few_shot(model, fs_query_loader, device)
            if args.wandb:
                wandb.log({'20-way 5-shot acc': acc, 'epoch': epoch})

        elif args.dataset == 'stl-10':
            stl10_train_loader, stl10_test_loader, _, _ = utils.get_data_loader('stl-10-eval', 256, False)
            annotation = utils.label_annotation(model, stl10_train_loader, args.n_classes, device)
            acc = utils.basic_node_evaluation(model, annotation, stl10_test_loader, device)
            if args.wandb:
                wandb.log({'Accuracy': acc, 'epoch': epoch})
            
        else:
            annotation = utils.label_annotation(model, train_loader, args.n_classes, device)
            acc = utils.basic_node_evaluation(model, annotation, test_loader, device)
            if args.wandb:
                wandb.log({'Accuracy': acc, 'epoch': epoch})

            if args.dataset == 'cifar-10': # try transfer learning
                cifar100_train_loader, cifar100_test_loader, _, _ = utils.get_data_loader('cifar-100', 256, False)
                annotation = utils.label_annotation(model, cifar100_train_loader, 100, device)
                acc = utils.basic_node_evaluation(model, annotation, cifar100_test_loader, device)
                if args.wandb:
                    wandb.log({'Transfer to CIFAR-100 accuracy': acc, 'epoch': epoch})

        
