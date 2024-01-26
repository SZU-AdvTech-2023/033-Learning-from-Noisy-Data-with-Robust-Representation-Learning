import os
import imp
from easydict import EasyDict
import algorithms as alg
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
import numpy as np
import dataloader.Dataloader_cifar_openset as dataloader
import dataloader.Dataloader_cifar as dataloader


args = {}
args['exp'] = 'cifar10_asym'
args['num_workers'] = 8
args['seed'] = 123
args['disp_step'] = 50
args['cuda'] = True
args['checkpoint'] = 10
args = EasyDict(args)

exp_config_file = os.path.join('.','config',args.exp+'.py')

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = os.path.join('.',config['exp_directory']) # the place where logs, models, and other stuff will be stored
config['checkpoint_dir'] = os.path.join('.',config['checkpoint_dir']) 
print("Loading experiment %s from file: %s" % (args.exp, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']

config['vis_dir'] = os.path.join('.', 'visualization', config['exp_dir'].split('/')[-1])
if (not os.path.isdir(config['vis_dir'])):
    os.makedirs(config['vis_dir'])
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if config['openset']:
    import dataloader.Dataloader_cifar_openset as dataloader
    loader = dataloader.cifar_dataloader(batch_size=data_train_opt['batch_size'],num_workers=args.num_workers,\
                                            ratio=config['noise_ratio'],root_dir=config['data_path'],\
                                            noise_file = config['noise_file'],open_noise=config['openset'])         
else:
    import dataloader.Dataloader_cifar as dataloader
    loader = dataloader.cifar_dataloader(batch_size=data_train_opt['batch_size'],num_workers=args.num_workers,\
                                            ratio=config['noise_ratio'],noise_mode = config['noise_mode'],\
                                            root_dir=config['data_path'],noise_file = config['noise_file'],\
                                            dataset = config['dataset'])                       
checkpoints = [str(i) for i in range(0, 201, 10)]
for ck in checkpoints:
    args['checkpoint'] = int(ck)
    config['disp_step'] = args.disp_step
    algorithm = getattr(alg, config['algorithm_type'])(config)
    if args.cuda: # enable cuda
        algorithm.load_to_gpu()
    if args.checkpoint > 0: # load checkpoint
        algorithm.load_checkpoint(args.checkpoint, train=False)

    train_loader,eval_loader,test_loader = loader.run()
    features = []
    labels = []
    clean_labels = []
    N = len(eval_loader.dataset)

    algorithm.networks['model'].eval()
    batch_size = train_loader.batch_size
    for i, batch in enumerate(train_loader):
        data = batch[0].cuda(non_blocking=True)
        target = batch[1].cuda(non_blocking=True)
        clean_target = batch[2].cuda(non_blocking=True)
        cur_features = algorithm.networks['model'](data, has_out=False)
        if i == 0:
            features = np.zeros((N, cur_features.shape[1]), dtype=np.float32)
            labels = np.zeros((N), dtype=np.int64)
            clean_labels = np.zeros((N), dtype=np.int64)
        if i < len(train_loader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = cur_features.cpu().detach().numpy()
            labels[i * batch_size: (i + 1) * batch_size] = batch[1].cpu().detach().numpy()
            clean_labels[i * batch_size: (i + 1) * batch_size] = batch[2].cpu().detach().numpy()
        else:
            # special treatment for final batch
            features[i * batch_size:] = cur_features.cpu().detach().numpy()
            labels[i * batch_size:] = batch[1].cpu().detach().numpy()
            clean_labels[i * batch_size:] = batch[2].cpu().detach().numpy()

    # clean_labels = eval_loader.dataset.clean_label

    # save features, labels and clean labels as npz file
    np.savez(os.path.join(config['vis_dir'], f"features_{args['checkpoint']}.npz"), features=features, labels=labels, clean_labels=clean_labels)