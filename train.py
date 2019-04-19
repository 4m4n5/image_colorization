# from transform import ReLabel, ToLabel, ToSP, Scale
from model import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from skimage import color

import time
import os
import sys
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt


""
class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


""
# Initialize parameters
args = {
    'path': '/scratch/as3ek/image_colorization/data/unsplash_cropped/',
    'dataset': 'unsplash',
    'batch_size': 16,
    'lr_G': 1e-4,
    'lr_D': 1e-4,
    'weight_decay': 0.9,
    'num_epoch': 100,
    'lamb': 100,
    'test': False, # 'path/to/model/for/test', 
    'model_G': False, # 'path/to/model/to/resume',
    'model_D': False, # 'path/to/model/to/resume',
    'plot': True,
    'save': True,
    'gpu': 0, 
    'image_size': 128
}

args = Arguments(args)

""
# Initialize models
# n_channels is input channels and n_classes is output channels
model_G = UNet(n_channels=3, n_classes=2)
model_D = ConvDis()

# Initialize start epochs for G and D
start_epoch_G = start_epoch_D = 0

# Start epoch for this session
start_epoch = 0

# Load saved models if resume training
if args.model_G:
    print('Resume model G: %s' % args.model_G)
    checkpoint_G = torch.load(model_G)
    model_G.load_state_dict(checkpoint_G['state_dict'])
    start_epoch_G = checkpoint_G['epoch']
    
if args.model_D:
    print('Resume model D: %s' % args.model_D)
    checkpoint_D = torch.load(model_D)
    model_D.load_state_dict(checkpoint_D['state_dict'])
    start_epoch_D = checkpoint_D['epoch']
    
assert start_epoch_G == start_epoch_D

# Shift models to GPU
model_G.cuda()
model_D.cuda()

# Initialize optimizers
optimizer_G = optim.Adam(model_G.parameters(),
                         lr=args.lr_G, betas=(0.5, 0.999),
                         eps=1e-8, weight_decay=args.weight_decay)
optimizer_D = optim.Adam(model_D.parameters(),
                         lr=args.lr_D, betas=(0.5, 0.999),
                         eps=1e-8, weight_decay=args.weight_decay)

# Load optimizers if resume training
if args.model_G:
    optimizer_G.load_state_dict(checkpoint_G['optimizer'])
if args.model_D:
    optimizer_D.load_state_dict(checkpoint_D['optimizer'])
    
# Loss Function
global criterion
criterion = nn.BCELoss()
global L1
L1 = nn.L1Loss()

""
# Dataset
data_root = args.path
dataset = args.dataset

if dataset == 'unsplash':
    from data_loader import Unsplash_Dataset as myDataset
# elif dataset == 'flower':
#     from load_data import Flower_Dataset as myDataset
# elif dataset == 'bob':
#     from load_data import Spongebob_Dataset as myDataset
else:
    raise ValueError('dataset type not supported')

# Define transform
image_transform = transforms.Compose([transforms.CenterCrop(args.image_size),
                                      transforms.ToTensor()])

data_train = myDataset(data_root, mode='train',
                       transform=image_transform,
                       types='raw',
                       shuffle=True
                      )

train_loader = data.DataLoader(data_train,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=4
                              )

data_val = myDataset(data_root, mode='test',
                     transform=image_transform,
                     types='raw',
                     shuffle=True
                    )

val_loader = data.DataLoader(data_val,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4
                            )

global val_bs
val_bs = val_loader.batch_size

""
# set up plotter, path, etc.
global iteration, print_interval, plotter, plotter_basic
iteration = 0
print_interval = 5
plotter = Plotter_GAN_TV()
plotter_basic = Plotter_GAN()

global img_path
size = str(args.image_size)
date = str(datetime.datetime.now().day) + '_' + str(datetime.datetime.now().day)
img_path = '/scratch/as3ek/image_colorization/results/img/%s/GAN_%s%s_%dL1_bs%d_%s_lr_D%s_lr_G%s/' \
           % (date, args.dataset, size, args.lamb, args.batch_size, 'Adam', str(args.lr_D), str(args.lr_G))
model_path = '/scratch/as3ek/image_colorization/results/model/%s/GAN_%s%s_%dL1_bs%d_%s_lr_D%s_lr_G%s/' \
           % (date, args.dataset, size, args.lamb, args.batch_size, 'Adam', str(args.lr_D), str(args.lr_G))

if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

""
import datetimeetime

""


""

