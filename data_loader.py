import os
import os.path as osp
import sys
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import scipy.io as io
import scipy.misc as misc
import glob
import csv
from skimage import color
import skimage

""
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


""
class Unsplash_Dataset(data.Dataset):
    def __init__(self, root, shuffle=False, mode='test', size=128, transform=None, 
                 target_transform=None, types='', show_ab=False, loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        
#         if large:
#             self.size = 480
#             self.imgpath = glob.glob(root + 'img_480/*.png')
#         else:
#             self.size = 224
#             self.imgpath = glob.glob(root + 'img/*.png')
        
        self.size = size
        
        self.types = types
        self.show_ab = show_ab # show ab channel in classify mode

        # read split
        self.train_file = set()
        self.test_file = set()
        
        self.path = []
        
        if mode == 'train':
            self.imgpath = glob.glob(root + 'train/*/*.jpg')
            for item in self.imgpath:
                self.path.append(item)
                    
        elif mode == 'test':
            self.imgpath = glob.glob(root + 'test/*/*.jpg')
            for item in self.imgpath:
                self.path.append(item)

        self.path = sorted(self.path)

        np.random.seed(0)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        if types == 'classify':
            ab_list = np.load('data/pts_in_hull.npy')
            self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ab_list)

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image
        img = np.array(img)
        
        # Resize image if necessary
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):
            img = skimage.transform.resize(img, (self.size, self.size))

        # Convert to lab space
        img_lab = color.rgb2lab(np.array(img)) # np array

        if self.types == 'classify':
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (self.size,self.size))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        # Normalize RGB images -1 to 1
        img = (img - 127.5) / 127.5
        # Rearrange channels RGB
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        
        # Rearrange channels LAB
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))
        # Normalize LAB images
        img_l = torch.unsqueeze(img_lab[0], 0) / 100. # L channel 0-100
        ######## CHANGED FROM 110 to 128 #########
        img_ab = (img_lab[1: : ] + 128) / 255. # ab channel -128 to 127

        if self.types == 'classify':
            if self.show_ab:
                return img_l, ab_class, img_ab
            return img_l, ab_class
        elif self.types == 'raw':
            return img_l, img_ab, img
            # if self.show_ab:
            #     return img_l, img_ab, None
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.path)


class CIFAR_Dataset(data.Dataset):
    def __init__(self, root, shuffle=False, mode='test', size=32, transform=None, 
                 target_transform=None, types='', show_ab=False, loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        if mode == 'test' and target_transform:
            self.image_transform = target_transform
        
        if mode == 'train':
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=self.image_transform)
        else:
            dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                   download=True, transform=self.image_transform)
        
        self.size = size
        self.types = types
        self.show_ab = show_ab # show ab channel in classify mode

        images = []
        
        for image, label in dataset:
            images.append(image)
        
        self.images = images
#         import pdb; pdb.set_trace()
        
        print('Load %d images, used %fs' % (len(images), time.time()-tic))

    def __getitem__(self, index):
        img = self.images[index]
#         img = np.array(img)
        img = np.transpose(img, (1, 2, 0))
        
        # Resize image if necessary
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):
            img = skimage.transform.resize(img, (self.size, self.size))

        # Convert to lab space
        img_lab = color.rgb2lab(np.array(img)) # np array

        if self.types == 'classify':
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (self.size,self.size))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        # Normalize RGB images -1 to 1
        img = (img - 127.5) / 127.5
        # Rearrange channels RGB
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        
        # Rearrange channels LAB
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))
        # Normalize LAB images
        img_l = torch.unsqueeze(img_lab[0], 0) / 100. # L channel 0-100 -> 0-1
        ######## CHANGED FROM 110 to 128 #########
        img_ab = (img_lab[1: : ] + 128) / 255. # ab channel -128 to 127 -> 0-1

        if self.types == 'classify':
            if self.show_ab:
                return img_l, ab_class, img_ab
            return img_l, ab_class
        elif self.types == 'raw':
            return img_l, img_ab, img
            # if self.show_ab:
            #     return img_l, img_ab, None
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.images)


""
if __name__ == '__main__':
    data_root = '/scratch/as3ek/image_colorization/data/unsplash_cropped/'
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    image_transform = transforms.Compose([
                              transforms.ToTensor(),
                          ])

    und = CIFAR_Dataset(data_root, mode='train', types='raw',
                      transform=image_transform)

    data_loader = data.DataLoader(und,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=4)

    for i, (data, target_ab, target_rgb) in enumerate(data_loader):
        print(i, data.size())
        break


