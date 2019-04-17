import os
import os.path as osp
import sys
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import scipy.io as io
import scipy.misc as misc
import glob
import csv
from skimage import color

##### HAVE TO ADD #####
# from transforms import ReLabel, ToLabel, ToSP, Scale 

""
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


""
class Unsplash_Dataset(data.Dataset):
    def __init__(self, root, shuffle=False, small=False, mode='test', transform=None, 
                 target_transform=None, types='', show_ab=False, large=False, loader=pil_loader):

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
        
        self.size = 128
        self.imgpath = glob.glob(root + '*/*.jpg')
        
        self.types = types
        self.show_ab = show_ab # show ab channel in classify mode

        # read split
        self.train_file = set()
        
        with open(self.root + 'train_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.train_file.add(str(row[0]).zfill(4))

        assert self.train_file.__len__() == 1392

        self.test_file = set()
        with open(self.root + 'test_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.test_file.add(str(row[0]).zfill(4))
        assert self.test_file.__len__() == 348

        self.path = []
        if mode == 'train':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.train_file:
                    self.path.append(item)
        elif mode == 'test':
            for item in self.imgpath:
                if item.split('/')[-1][6:6+4] in self.test_file:
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
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):
            img = misc.imresize(img, (self.size, self.size))

        img_lab = color.rgb2lab(np.array(img)) # np array
        # img_lab = img_lab[13:13+224, 13:13+224, :]

        if self.types == 'classify':
            X_a = np.ravel(img_lab[:,:,1])
            X_b = np.ravel(img_lab[:,:,2])
            img_ab = np.vstack((X_a, X_b)).T
            _, ind = self.nbrs.kneighbors(img_ab)
            ab_class = np.reshape(ind, (self.size,self.size))
            # print(ab_class.shape, ab_class.dtype, np.amax(ab_class), np.amin(ab_class))
            ab_class = torch.unsqueeze(torch.LongTensor(ab_class), 0)

        img = (img - 127.5) / 127.5 # -1 to 1
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100
        img_ab = (img_lab[1::] + 0) / 110. # ab channel -110 - 110

        if self.types == 'classify':
            if self.show_ab:
                return img_l, ab_class, img_ab
            return img_l, ab_class
        elif self.types == 'raw':
            return img_l, img
            # if self.show_ab:
            #     return img_l, img_ab, None
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.path)


""
glob.glob('data/unsplash_cropped_resized/' + '*/*.jpg')[0].zfill(4)

""

