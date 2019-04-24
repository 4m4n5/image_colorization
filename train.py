# from transform import ReLabel, ToLabel, ToSP, Scale
from model import *
from utils import *
from loss import *

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
import datetime
import warnings
warnings.filterwarnings('ignore')


class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


# Initialize parameters
args = {
    'path': '/scratch/as3ek/image_colorization/data/unsplash_cropped/',
    'dataset': 'cifar',
    'batch_size': 32,
    'lr_G': 1e-4,
    'lr_D': 5e-4,
    'weight_decay': 0.0,
    'num_epoch': 100,
    'lamb': 100,
    'test': False, # 'path/to/model/for/test', 
    'model_G': False, # 'path/to/model/to/resume',
    'model_D': False, # 'path/to/model/to/resume',
    'plot': True,
    'save': True,
    'gpu': 0, 
    'image_size': 32
}

args = Arguments(args)


def main(args):
    # Initialize models
    # n_channels is input channels and n_classes is output channels
    model_G = UNet(n_channels=1, n_classes=3)
    model_D = ConvDis(in_channels=3, in_size=args.image_size)

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
    global FeatureLoss
    FeatureLoss = FeatureLoss()

    # Dataset
    data_root = args.path
    dataset = args.dataset

    if dataset == 'unsplash':
        from data_loader import Unsplash_Dataset as myDataset
    elif dataset == 'cifar':
        from data_loader import CIFAR_Dataset as myDataset
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
                                  shuffle=False
                                  )

    data_val = myDataset(data_root, mode='test',
                         transform=image_transform,
                         types='raw',
                         shuffle=True
                        )

    val_loader = data.DataLoader(data_val,
                                 batch_size=args.batch_size,
                                 shuffle=False
                                )

    global val_bs
    val_bs = val_loader.batch_size

    # set up plotter, path, etc.
    global iteration, print_interval, plotter, plotter_basic, plot_train_result_interval
    iteration = 0
    print_interval = 5
    plot_train_result_interval = 100
    plotter = Plotter_GAN_TV()
    plotter_basic = Plotter_GAN()

    global img_path
    size = str(args.image_size)
    date = str(datetime.datetime.now().month) + '_' + str(datetime.datetime.now().day)
    img_path = '/scratch/as3ek/image_colorization/results/img/%s/GAN_%s%s_%dL1_bs%d_%s_lr_D%s_lr_G%s/' \
               % (date, args.dataset, size, args.lamb, args.batch_size, 'Adam', str(args.lr_D), str(args.lr_G))
    model_path = '/scratch/as3ek/image_colorization/results/model/%s/GAN_%s%s_%dL1_bs%d_%s_lr_D%s_lr_G%s/' \
               % (date, args.dataset, size, args.lamb, args.batch_size, 'Adam', str(args.lr_D), str(args.lr_G))

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    start_epoch = 0

    for epoch in range(start_epoch, args.num_epoch):
        print('Epoch {}/{}'.format(epoch, args.num_epoch - 1))
        print('-' * 20)
#         if epoch == 0:
#             val_lerrG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch=-1)
        # train
        train_errG, train_errD = train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, iteration)
        # validate
        val_lerrG, val_errD = validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch)

        plotter.train_update(train_errG, train_errD)
        plotter.val_update(val_lerrG, val_errD)
        plotter.draw(img_path + 'train_val.png')

        if args.save:
            print('Saving check point')
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_G.state_dict(),
                             'optimizer': optimizer_G.state_dict(),
                             },
                             filename=model_path+'G_epoch%d.pth.tar' \
                             % epoch)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model_D.state_dict(),
                             'optimizer': optimizer_D.state_dict(),
                             },
                             filename=model_path+'D_epoch%d.pth.tar' \
                             % epoch)


def train(train_loader, model_G, model_D, optimizer_G, optimizer_D, epoch, iteration):
    errorG = AverageMeter() # will be reset after each epoch
    errorD = AverageMeter() # will be reset after each epoch
    errorG_basic = AverageMeter() # basic will be reset after each print
    errorD_basic = AverageMeter() # basic will be reset after each print
    errorD_real = AverageMeter()
    errorD_fake = AverageMeter()
    errorG_GAN = AverageMeter()
    errorG_R = AverageMeter()

    model_G.train()
    model_D.train()

    real_label = 1
    fake_label = 0

    for i, (data, target_ab, target_rgb) in enumerate(train_loader):
        data, target_ab, target_rgb = Variable(data.cuda()), Variable(target_ab.cuda()), Variable(target_rgb.cuda())

        ########################
        # Update D network
        ########################
        
        # Train with real
        model_D.zero_grad()
        output = model_D(target_rgb)
        label = torch.FloatTensor(target_rgb.size(0)).fill_(real_label).cuda()
        labelv = Variable(label)
        errD_real = criterion(torch.squeeze(output), labelv)
        errD_real.backward()
        D_x = output.data.mean() 

        # Train with fake
        # Generate fake output from the Generator
        fake =  model_G(data)
        labelv = Variable(label.fill_(fake_label))
        output = model_D(fake.detach())
        errD_fake = criterion(torch.squeeze(output), labelv)
        errD_fake.backward()
        D_G_x1 = output.data.mean()

        errD = errD_real + errD_fake
        optimizer_D.step()

        ########################
        # Update G network
        ########################
        model_G.zero_grad()
        labelv = Variable(label.fill_(real_label))
        output = model_D(fake)
        
#         real_rgb = target_rgb
#         fake_rgb = fake
        
        errG_GAN = criterion(torch.squeeze(output), labelv)
        errG_L1 = L1(fake.view(fake.size(0),-1), target_rgb.view(target_rgb.size(0),-1))
        errG_Feature = FeatureLoss(fake, target_rgb)

        errG = errG_GAN + args.lamb * errG_Feature
        errG.backward()
        D_G_x2 = output.data.mean()
        optimizer_G.step()

        # store error values
        errorG.update(errG.data.item(), target_ab.size(0), history=1)
        errorD.update(errD.data.item(), target_ab.size(0), history=1)
        errorG_basic.update(errG.data.item(), target_ab.size(0), history=1)
        errorD_basic.update(errD.data.item(), target_ab.size(0), history=1)
        errorD_real.update(errD_real.data.item(), target_ab.size(0), history=1)
        errorD_fake.update(errD_fake.data.item(), target_ab.size(0), history=1)

        errorD_real.update(errD_real.data.item(), target_ab.size(0), history=1)
        errorD_fake.update(errD_fake.data.item(), target_ab.size(0), history=1)
        errorG_GAN.update(errG_GAN.data.item(), target_ab.size(0), history=1)
        errorG_R.update(errG_L1.data.item(), target_ab.size(0), history=1)


        if iteration % print_interval == 0:
            print('Epoch%d[%d/%d]: Loss_D: %.4f(R%0.4f+F%0.4f) Loss_G: %0.4f(GAN%.4f+R%0.4f) D(x): %.4f D(G(z)): %.4f / %.4f' \
                % (epoch, i, len(train_loader),
                errorD_basic.avg, errorD_real.avg, errorD_fake.avg,
                errorG_basic.avg, errorG_GAN.avg, errorG_R.avg,
                D_x, D_G_x1, D_G_x2
                ))
            # plot image
            plotter_basic.g_update(errorG_basic.avg)
            plotter_basic.d_update(errorD_basic.avg)
            plotter_basic.draw(img_path + 'train_basic.png')
            # reset AverageMeter
            errorG_basic.reset()
            errorD_basic.reset()
            errorD_real.reset()
            errorD_fake.reset()
            errorG_GAN.reset()
            errorG_R.reset()
        
        if iteration % plot_train_result_interval == 0:
            vis_result_rgb(data.data, fake.data, target_rgb.data, epoch, True)

        iteration += 1

    return errorG.avg, errorD.avg


def validate(val_loader, model_G, model_D, optimizer_G, optimizer_D, epoch):
    errorG = AverageMeter()
    errorD = AverageMeter()

    model_G.eval()
    model_D.eval()

    real_label = 1
    fake_label = 0

    for i, (data, target_ab, target_rgb) in enumerate(val_loader):
        data, target_ab, target_rgb = Variable(data.cuda()), Variable(target_ab.cuda()), Variable(target_rgb.cuda())
        
        ########################
        # D network
        ########################
        
         # Train with real
        output = model_D(target_rgb)
        label = torch.FloatTensor(target_rgb.size(0)).fill_(real_label).cuda()
        labelv = Variable(label)
        errD_real = criterion(torch.squeeze(output), labelv)
        
        # Train with fake
        # Generate fake output from the Generator
        fake =  model_G(data)
        labelv = Variable(label.fill_(fake_label))
        output = model_D(fake.detach())
        errD_fake = criterion(torch.squeeze(output), labelv)
       
        errD = errD_real + errD_fake
        
        
        ########################
        # G network
        ########################
        labelv = Variable(label.fill_(real_label))
        output = model_D(fake)
        errG_GAN = criterion(torch.squeeze(output), labelv)
#         errG_L1 = L1(fake.view(fake.size(0),-1), target_ab.view(target_ab.size(0),-1))
        errG_Feature = FeatureLoss(fake, target_rgb)
    
        errG = errG_GAN + args.lamb * errG_Feature

        errorG.update(errG.data.item(), target_ab.size(0), history=1)
        errorD.update(errD.data.item(), target_ab.size(0), history=1)
        
        
        if i == 0:
            vis_result_rgb(data.data, fake.data, target_rgb.data, epoch)

        if i % 50 == 0:
            print('Validating Epoch %d: [%d/%d]' \
                % (epoch, i, len(val_loader)))

    print('Validation: Loss_D: %.4f Loss_G: %.4f '\
        % (errorD.avg, errorG.avg))

    return errorG.avg, errorD.avg


def lab_to_rgb(l, ab):
    rgb = torch.zeros(16, 3, 32, 32)
    lab = torch.cat([l, ab], dim=1)
    print(lab.size())

    for i in range(lab.size(0)):
        lab_img = lab[i]
        lab_img = lab_img.permute(1, 2, 0)
        rgb_img = torch.tensor(color.lab2rgb(lab_img))
        rgb_img = rgb_img.permute(2, 0, 1)

        rgb[i] = rgb_img

    return rgb


def vis_result_rgb(data_l, fake_rgb, target_rgb, epoch, is_train=False):
    '''visualize images for GAN'''
    img_list = []
#     o_ab_list = []
    for i in range(min(32, val_bs)):
        l = torch.unsqueeze(torch.squeeze(data_l[i]), 0).cpu().numpy()
        fake_img = fake_rgb[i].cpu().numpy()
        target_img = target_rgb[i].cpu().numpy()
        
        target_img = (target_img + 1) / 2.
        fake_img = (fake_img + 1) / 2.
        
        
        raw_rgb = np.array(np.transpose(target_img, (1,2,0))) 
        pred_rgb = np.array(np.transpose(fake_img, (1,2,0)))
        
        
        
        grey = np.transpose(l, (1,2,0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)
        img_list.append(np.concatenate((grey, raw_rgb, pred_rgb), 1))

    img_list = [np.concatenate(img_list[4*i:4*(i+1)], axis=1) for i in range(len(img_list) // 4)]
    img_list = np.concatenate(img_list, axis=0)

    plt.figure(figsize=(36,27))
    plt.imshow(img_list)
    plt.axis('off')
    plt.tight_layout()
    if is_train:
        plt.savefig(img_path + 'epoch%d_train.png' % epoch)
    else:
        plt.savefig(img_path + 'epoch%d_val.png' % epoch)
    plt.clf()


def vis_result(data_l, target_ab, output_ab, epoch, is_train=False):
    '''visualize images for GAN'''
    img_list = []
#     o_ab_list = []
    for i in range(min(32, val_bs)):
        l = torch.unsqueeze(torch.squeeze(data_l[i]), 0).cpu().numpy()
        t_ab = target_ab[i].cpu().numpy()
        o_ab = output_ab[i].cpu().numpy()
        
        t_ab = (t_ab * 255.) - 128 
        o_ab = (o_ab * 255.) - 128 
        
#         o_ab_list.append(o_ab)
        
        t_l = l * 100.
        
        t_lab = np.concatenate((t_l, t_ab), axis=0)
        o_lab = np.concatenate((t_l, o_ab), axis=0)
        
        raw_rgb = color.lab2rgb(np.array(np.transpose(t_lab, (1,2,0))))
        pred_rgb = color.lab2rgb(np.array(np.transpose(o_lab, (1,2,0))))

        grey = np.transpose(l, (1,2,0))
        grey = np.repeat(grey, 3, axis=2).astype(np.float64)
        img_list.append(np.concatenate((grey, raw_rgb, pred_rgb), 1))

    img_list = [np.concatenate(img_list[4*i:4*(i+1)], axis=1) for i in range(len(img_list) // 4)]
    img_list = np.concatenate(img_list, axis=0)

    plt.figure(figsize=(36,27))
    plt.imshow(img_list)
    plt.axis('off')
    plt.tight_layout()
    if is_train:
        plt.savefig(img_path + 'epoch%d_train.png' % epoch)
    else:
        plt.savefig(img_path + 'epoch%d_val.png' % epoch)
    plt.clf()


main(args)


