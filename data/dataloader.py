import json
import os.path as osp
import time
import requests
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from BCL.randaug import *
import os
from PIL import Image
import io
import logging
logger = logging.getLogger('global')


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img


# Image statistics
RGB_statistics = {
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'clip': {
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711]
    }
}

def calculate_momentum_weight(momentum_loss, epoch):
    
    momentum_weight = ((momentum_loss[epoch-1]-torch.mean(momentum_loss[epoch-1,:]))/torch.std(momentum_loss[epoch-1,:]))
    momentum_weight = ((momentum_weight/torch.max(torch.abs(momentum_weight[:])))/2+1/2).detach().cpu().numpy()

    return momentum_weight


def get_data_ransform_memoboosted(split, rgb_mean, rbg_std,
                                  rand_k, min_strength, rand_strength,  key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(
                size=224, scale=(0.5, 1), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0)]),

            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(
                size=224, scale=(0.5, 1), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]

# Dataset


class LT_Dataset(Dataset):

    def __init__(self, root, txt, stage=1, rank_k=1, rand_strength=0, transform=None, mode="train"):
        self.img_path = []
        self.labels = []
        self.transform = transform
        class_num = 100
        self.idxsNumPerClass = [0] * class_num
        idx_num = 0
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                label = int(line.split()[1])
                self.labels.append(label)
                self.idxsNumPerClass[label] += 1
                idx_num += 1
        self.momentum_weight = np.empty(idx_num)
        self.momentum_weight[:] = 0
        self.stage = stage
        self.rank_k = rank_k
        self.rand_strength = rand_strength
        
    
    def update_momentum_weight(self, momentum_loss, epoch):
        momentum_weight_norm = calculate_momentum_weight(momentum_loss, epoch)
        self.momentum_weight = momentum_weight_norm

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        # if self.stage == 1:

        #     if self.transform is not None:
        #         sample = self.transform(sample)

        #     return sample, label, index
        
        if self.stage == 2 and self.mode == "train":

            if self.rank_k == 1:
                min_strength = 10 # training stability
                memo_boosted_aug = transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=3),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        RandAugment_prob(self.rand_k, min_strength + (self.args.rand_strength - min_strength)*self.momentum_weight[idx], 1.0*self.momentum_weight[idx]),
                        transforms.ToTensor(),
                    ])
            else:
                min_strength = 5 # training stability
                memo_boosted_aug = transforms.Compose([
                        transforms.RandomResizedCrop(32, scale=(0.1, 1.0), interpolation=3),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        RandAugment_prob(self.rand_k, min_strength + (self.args.rand_strength - min_strength)*self.momentum_weight[idx]*np.random.rand(1), 1.0*self.momentum_weight[idx]),
                        transforms.ToTensor(),
                    ])
            sample = memo_boosted_aug(sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        return sample, label, index

# Load datasets


def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, stage=1, rank_k=1, rand_strength=0):

    if phase == 'train_plain':
        txt_split = 'train'
    elif phase == 'train_val':
        txt_split = 'val'
        phase = 'train'
    else:
        txt_split = phase
    txt = './data/%s/%s_%s.txt' % (dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    key = 'clip'
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase not in ['train', 'val']:
        transform = get_data_transform('test', rgb_mean, rgb_std, key)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key)

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, stage, rank_k, rand_strength, transform)
    print(len(set_))
    if phase == 'test' and test_open:
        open_txt = './data/%s/%s_open.txt' % (dataset, dataset)
        print('Testing with opensets from %s' % (open_txt))
        open_set_ = LT_Dataset('./data/%s/%s_open' %
                               (dataset, dataset), open_txt, transform)
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                          sampler=sampler_dic['sampler'](
                              set_, **sampler_dic['params']),
                          num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
