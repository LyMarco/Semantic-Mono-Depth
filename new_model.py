# -*- coding: utf-8 -*-
"""Copy of [Clean] Model 3 Monodepth-seg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ww5xG8wV8ZtS3sU1PRO0PkRt5-HZ4WkC

#Setup
"""

# Commented out IPython magic to ensure Python compatibility.
# # Look for Tesla P100-PCIE
# %%shell
# nvidia-smi

import os
import torch
import time
import random

import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, transform, color, data, img_as_float
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageEnhance
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""##Building datasets"""

# Download segment data
# !wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip
# !unzip data_semantics.zip training/semantic/* -d ./images
# !unzip data_semantics.zip training/image_2/* -d ./images

# # Download KITTI 2015 Stereo dataset
# !wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
# !unzip data_scene_flow.zip training/disp_noc_0/* -d ./images/
# !unzip data_scene_flow.zip training/disp_noc_1/* -d ./images/
# !unzip data_scene_flow.zip training/disp_occ_0/* -d ./images/
# !unzip data_scene_flow.zip training/disp_occ_1/* -d ./images/
# !unzip data_scene_flow.zip training/image_3/* -d ./images/

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# 
# cd 'images/training/image_3'
# 
# for name in *_11.png
# do
#     rm $name
# done

"""## Data Augmentation"""

class TransformMap(object):
    def __init__(self, transform, input_only=True):
        self.transform = transform
        self.input_only = input_only
    
    def __call__(self, sample, *args, **kwargs):
        left, right, disp_l, disp_r, semantic = sample['left'], \
        sample['right'], sample['disp_left'], \
        sample['disp_right'], sample['semantic']

        left = self.transform(left, *args, **kwargs)
        right = self.transform(right, *args, **kwargs)

        if not self.input_only:     
            disp_l = self.transform(disp_l, *args, **kwargs)
            disp_r = self.transform(disp_r, *args, **kwargs)
            semantic = self.transform(semantic, *args, **kwargs)

        return {'left': left, 'right': right, 'disp_left':disp_l, \
                'disp_right':disp_r, 'semantic':semantic}

# Data Transformations
# Rescale object repurposed from Pytorch tutorial on Datasets by Sasank Chilamkurthy
class RescaleDispSeg(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        left, right, disp_l, disp_r, semantic = sample['left'], sample['right'], sample['disp_left'], sample['disp_right'], sample['semantic']

        h, w = left.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        left = transform.resize(left, (new_h, new_w))
        right = transform.resize(right, (new_h, new_w))
        disp_l = transform.resize(disp_l, (new_h, new_w))
        disp_r = transform.resize(disp_r, (new_h, new_w))
        semantic = transform.resize(semantic, (new_h, new_w))

        return {'left': left, 'right': right, 'disp_left':disp_l, 'disp_right':disp_r, 'semantic':semantic}

class TransformIdentity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.clip(sample * 1.0, 0, 255)

class TransformRandom(object):
    def __init__(self, transform, min, max, n=1):
        self.transform = transform
        self.min = min
        self.max = max
        self.n = n

    def __call__(self, sample):
        random = np.random.uniform(self.min, self.max, self.n)
        sample = self.transform(sample, random)
        return sample

class TransformNoise(object):
    def __init__(self, mean=0, std=0):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        w, h = sample.shape[:2]
        noise = np.random.normal(loc=self.mean, scale=self.std, size=(w, h, 3))
        sample = np.clip(sample + noise, 0, 255)
        return sample

class TransformGamma(object):
    def __init__(self, min=0.8, max=1.2):
        self.min = min
        self.max = max
    
    def __call__(self, sample, random):
        gamma_shift = random
        sample = np.clip(sample ** gamma_shift, 0, 255)
        return sample

class TransformBrightness(object):
    def __init__(self, min=0.5, max=2.0):
        self.min = min
        self.max = max
    
    def __call__(self, sample, random):
        bright_shift = random
        sample = np.clip(sample * bright_shift, 0, 255)
        return sample

class TransformColor(object):
    def __init__(self, min=0.8, max=1.2):
        self.min = min
        self.max = max
    
    def __call__(self, sample, random):
        colours = random
        white = np.ones((sample.shape[0], sample.shape[1]))
        colour_image = np.stack([white * colours[i] for i in range(3)], axis=2)

        sample = np.clip(sample * colour_image, 0, 255)
        return sample

AUG_transform = transforms.Compose([
    transforms.RandomChoice([
        TransformMap(TransformIdentity()),
        TransformMap(TransformNoise(0, 30)),
    ]),
    transforms.RandomChoice([
        TransformMap(TransformIdentity()),
        TransformRandom(TransformMap(TransformBrightness()), 0.5, 1.5, 1)
    ]),
    transforms.RandomChoice([
        TransformMap(TransformIdentity()),
        TransformRandom(TransformMap(TransformColor()), 0.8, 1.2, 3)
    ]),
    transforms.RandomChoice([
        TransformMap(TransformIdentity()),
        TransformRandom(TransformMap(TransformGamma()), 0.8, 1.2, 1)
    ]),
    RescaleDispSeg((256, 512)),
    TransformMap(transforms.ToTensor(), input_only=False),
    TransformMap(transforms.Lambda(lambda x: x.to(device)), input_only=False),
])

Normal_transform = transforms.Compose([
    RescaleDispSeg((256, 512)),
    TransformMap(transforms.ToTensor(), input_only=False),
    TransformMap(transforms.Lambda(lambda x: x.to(device)), input_only=False)
])

"""##Load Dataset"""

class DispSegDataset(Dataset):
    """Dataset for left-right images, disparity and segmentation maps."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with folders 'input' and 'depth_maps'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.left_dir = os.path.join(self.root_dir, 'image_2/*')
        self.right_dir = os.path.join(self.root_dir, 'image_3/*')
        self.disp_left_dir = os.path.join(self.root_dir, 'disp_occ_0/*')
        self.disp_right_dir = os.path.join(self.root_dir, 'disp_occ_1/*')
        self.semantic_dir = os.path.join(self.root_dir, 'semantic/*')

        self.left = io.imread_collection(self.left_dir)
        self.right = io.imread_collection(self.right_dir)
        self.disp_left = io.imread_collection(self.disp_left_dir)
        self.disp_right = io.imread_collection(self.disp_right_dir)
        self.semantic = io.imread_collection(self.semantic_dir)

    def __len__(self):
      # return len(self.left)
      return 199
      
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = {'left': self.left[idx], 'right': self.right[idx], 
                'disp_left': self.disp_left[idx], 'disp_right': self.disp_right[idx],
                'semantic': self.semantic[idx]}

        if self.transform:
            item = self.transform(item)    
        return item

# Load the datasest
# Original scale is 1242x375, rescaled to 512x256
data_path = "./images/"

train_dir = os.path.join(data_path, 'training')
# test_dir = os.path.join(data_path, 'testing')

depth_train_full = DispSegDataset(train_dir, AUG_transform)

def cpu_img(image, n=1):
  if n == 1:
      return image[0].cpu()
  img = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))
  if np.max(img) > 1:
      return img.astype(np.uint8)
  return img

# Test the dataset is working
# After data augmentation, the input images should be [0, 255] while the 
# groundtruth images should be [0, 1]. This should be learnable by the model

print(len(depth_train_full))
k = 1
plt.figure(figsize=(16, 16))
image = depth_train_full[k]
plt.subplot(3, 2, 1), plt.imshow(cpu_img(image['left'], n=3))
plt.subplot(3, 2, 2), plt.imshow(cpu_img(image['right'], n=3))
plt.subplot(3, 2, 3), plt.imshow(cpu_img(image['disp_left']))
plt.subplot(3, 2, 4), plt.imshow(cpu_img(image['disp_right']))
plt.subplot(3, 2, 5), plt.imshow(cpu_img(image['semantic']))

"""#Model

## Model 1
"""

class CCNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CCNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, padding=padding)
        std0 = np.sqrt(2/(kernel_size ** 2 * in_channels))
        nn.init.normal_(self.conv0.weight, std=std0)
        self.batch0 = nn.BatchNorm2d(out_channels)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, padding=padding)
        std1 = np.sqrt(2/(kernel_size ** 2 * out_channels))
        nn.init.normal_(self.conv1.weight, std=std1)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.batch0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        return x

class CTNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, stride=2,
                 dropout=False):
        super(CTNet, self).__init__()
        
        self.conv0 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        padding=padding, stride=stride)
        std0 = np.sqrt(2/(kernel_size ** 2 * in_channels))
        nn.init.normal_(self.conv0.weight, std=std0)
        self.batch0 = nn.BatchNorm2d(out_channels)
        if dropout:
            self.dropout = nn.Dropout2d()

    def forward(self, x, x1):
        x = self.conv0(x)
        x = self.batch0(x)
        # _, _, w, h = x.size()
        _, _, w0, h0 = x.size()
        _, _, w1, h1 = x1.size()
        dw, dh = w1 - w0, h1 - h0
        lpad = dw // 2
        rpad = dw - lpad
        tpad = dh // 2
        bpad = dh - tpad
        x = F.pad(x, (lpad, rpad, tpad, bpad))
        x = torch.cat([x, x1], dim=1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x

class SegNet(nn.Module):
    def __init__(self, n=3, i=4, dropout=False, num_classes=6):
        super(SegNet, self).__init__()
        self.conv0 = CCNet(n, 2**(i+0))
        self.pool0 = nn.MaxPool2d(2)
        self.conv1 = CCNet(2**(i+0), 2**(i+1))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = CCNet(2**(i+1), 2**(i+2))
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = CCNet(2**(i+2), 2**(i+3))
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = CCNet(2**(i+3), 2**(i+4))
        self.upco5 = CTNet(2**(i+4), 2**(i+3), dropout=dropout)
        self.conv5 = CCNet(2**(i+4), 2**(i+3))
        self.upco6 = CTNet(2**(i+3), 2**(i+2), dropout=dropout)
        self.conv6 = CCNet(2**(i+3), 2**(i+2))
        self.upco7 = CTNet(2**(i+2), 2**(i+1), dropout=dropout)
        self.conv7 = CCNet(2**(i+2), 2**(i+1))
        self.upco8 = CTNet(2**(i+1), 2**(i+0), dropout=dropout)
        self.conv8 = CCNet(2**(i+1), 2**(i+0))
        self.conv = nn.Conv2d(2**(i+0), n, 1)

        self.disp = nn.Sequential(
            nn.Conv2d(n, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),)

        self.seg = nn.Conv2d(n, num_classes, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Encoder
        x = x0 = self.conv0(x)
        x = self.pool0(x)
        x = x1 = self.conv1(x)
        x = self.pool1(x)
        x = x2 = self.conv2(x)
        x = self.pool2(x)
        x = x3 = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)

        # Disparity Decoder
        disp = self.upco5(x, x3)
        disp = self.conv5(disp)
        disp = self.upco6(disp, x2)
        disp = self.conv6(disp)
        disp = self.upco7(disp, x1)
        disp = self.conv7(disp)
        disp = self.upco8(disp, x0)
        disp = self.conv8(disp)
        disp = self.conv(disp)
        disp = self.disp(disp)

        # Segmentation decoder
        seg = self.upco5(x, x3)
        seg = self.conv5(seg)
        seg = self.upco6(seg, x2)
        seg = self.conv6(seg)
        seg = self.upco7(seg, x1)
        seg = self.conv7(seg)
        seg = self.upco8(seg, x0)
        seg = self.conv8(seg)
        seg = self.conv(seg)
        seg = self.seg(seg)

        return disp, seg

"""## Model 2"""

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, num_features, num_seg_classes):
        super(UNet, self).__init__()
        self.encode1 = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )
        self.encode2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU())
        self.encode3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
            nn.Conv2d(num_features * 4, num_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
        )
        self.encode4 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(),
            nn.Conv2d(num_features * 8, num_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(),
        )

        self.encode5 = nn.Sequential(
            nn.Conv2d(num_features * 8, num_features * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*16),
            nn.ReLU(),
            nn.Conv2d(num_features * 16, num_features * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*16),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 16, num_features * 8, kernel_size=2, stride=2)
        )

        self.decode1D = nn.Sequential(
            nn.Conv2d(num_features * 16, num_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.Conv2d(num_features * 8, num_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 8, num_features * 4, kernel_size=2, stride=2)
        )
        self.decode2D = nn.Sequential(
            nn.Conv2d(num_features * 8, num_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.Conv2d(num_features * 4, num_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        )
        self.decode3D = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2)
        )
        self.decode4D = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(), 
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
        )
                
        self.disp = nn.Sequential(
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),)
        

        self.decode1S = nn.Sequential(
            nn.Conv2d(num_features * 16, num_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.Conv2d(num_features * 8, num_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 8, num_features * 4, kernel_size=2, stride=2)
        )
        self.decode2S = nn.Sequential(
            nn.Conv2d(num_features * 8, num_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.Conv2d(num_features * 4, num_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        )
        self.decode3S = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2)
        )
        self.decode4S = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(), 
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
        )
        self.seg = nn.Conv2d(num_features, num_seg_classes, kernel_size=kernel, padding=1)



    def forward(self, x):
        # Encoder
        encode1 = self.encode1(x)
        encode1_out = F.max_pool2d(encode1, kernel_size=2)
        
        encode2 = self.encode2(encode1_out)
        encode2_out = F.max_pool2d(encode2, kernel_size=2)


        encode3 = self.encode3(encode2_out)
        encode3_out = F.max_pool2d(encode3, kernel_size=2)

        encode4 = self.encode4(encode3_out)
        encode4_out = F.max_pool2d(encode4, kernel_size=2)

        encode5_out = self.encode5(encode4_out)

        # Decoder Depth
        decode1_input = torch.cat([encode4, encode5_out], dim=1)
        decode1 =  self.decode1D(decode1_input)

        decode2_input = torch.cat([decode1, encode3], dim=1)
        decode2 =  self.decode2D(decode2_input)

        decode3_input = torch.cat([decode2, encode2], dim=1)
        decode3 =  self.decode3D(decode3_input)
        
        decode4_input = torch.cat([decode3, encode1], dim=1)
        decode4 =  self.decode4D(decode4_input)
        self.out_disp = self.disp(decode4)

        # Decoder semantic
        decode1_input_sem = torch.cat([encode4, encode5_out], dim=1)
        decode1_sem =  self.decode1S(decode1_input_sem)

        decode2_input_sem = torch.cat([decode1_sem, encode3], dim=1)
        decode2_sem =  self.decode2S(decode2_input_sem)

        decode3_input_sem = torch.cat([decode2_sem, encode2], dim=1)
        decode3_sem =  self.decode3S(decode3_input_sem)
        
        decode4_input_sem = torch.cat([decode3_sem, encode1], dim=1)
        decode4_sem =  self.decode4S(decode4_input_sem)
        self.out_seg = self.seg(decode4_sem)

        return self.out_disp, self.out_seg

"""## Model Details"""

# model = SegNet()
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)

# model = UNet(in_channels=3, out_channels=1, kernel=3, num_features=16, num_seg_classes=6)
# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)

"""# Training

## Losses
"""

# Loss Functions

a_d = 1.0
a_s = 0.1
gamma = 0.85

def SSIM_loss(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 **2 

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def depth_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    # SSIM = pytorch_ssim.SSIM(window_size=11)
    L1 = nn.L1Loss(size_average=False, reduce=False)

    sim_left = torch.mean(SSIM_loss(out_disp, target_disp_l))
    sim_right = torch.mean(SSIM_loss(out_disp, target_disp_r))

    pixel_loss_l = torch.mean(L1(out_disp, target_disp_l))
    pixel_loss_r = torch.mean(L1(out_disp, target_disp_r))

    left_loss = gamma * sim_left + (1- gamma) * pixel_loss_l
    right_loss = gamma * sim_right + (1- gamma) * pixel_loss_r

    return left_loss + right_loss


def sem_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    CE = nn.CrossEntropyLoss()
    return CE(out_sem, target_sem.long().squeeze(1)) 


def total_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    d_ap = depth_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem)
    d_sem = sem_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem)
    return a_d * d_ap + a_s * d_sem


def bce_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    BCE = nn.BCELoss()
    return BCE(out_disp, target_disp_l)


def L1_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    L1 = nn.L1Loss(size_average=False, reduce=False)
    return torch.mean(L1(out_disp, target_disp_l))

def validation(model, criterion, testdata, device):
    
    sample = testdata[199]

    left = torch.unsqueeze(sample['left'], 0)
    disp_left = torch.unsqueeze(sample['disp_left'], 0)
    disp_right = torch.unsqueeze(sample['disp_right'], 0)
    semantic = torch.unsqueeze(sample['semantic'], 0)
    
    model_disp, model_semantic = model(left)
        
    loss = criterion(model_disp, model_semantic, disp_left, disp_right, semantic)
    return loss.data.item()

    # losses = []

    # for i_batch, test_sample in enumerate(testdata):
    #     left = test_sample['left']
    #     right = test_sample['right']
    #     disp_left = test_sample['disp_left']
    #     disp_right = test_sample['disp_right']
    #     semantic = test_sample['semantic']
        
    #     model_disp, model_semantic = model(left)
            
    #     loss = criterion(model_disp, model_semantic, disp_left, disp_right, semantic)
    #     losses.append(loss.data.item())

    # val_loss = np.mean(losses)
    # return val_loss

"""## Main training loop"""

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# Main training loop

def train(args):
    torch.set_num_threads(5)

    np.random.seed(args.seed)

    # DATALOADERS
    trainloader = torch.utils.data.DataLoader(args.train_data, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(args.test_data, batch_size=args.batch_size, shuffle=True)
    
    # MODEL
    num_in_channels = 1 if not args.downsize_input else 3
    if args.model == "SegNet":
        model = SegNet()
    elif args.model == "UNet":
        model = UNet(in_channels=num_in_channels, out_channels=1, kernel=args.kernel,
                       num_features=args.num_filters, num_seg_classes=args.num_sem_classes)
    else:
        model = UNet(in_channels=num_in_channels, out_channels=1, kernel=args.kernel,
                       num_features=args.num_filters, num_seg_classes=args.num_sem_classes)


    # LOSS FUNCTION
    if args.loss == "DEPTH-SEG":
        criterion = total_loss
    elif args.loss == "DEPTH":
        criterion = depth_loss
    elif args.loss == "SEG":
        criterion = sem_loss
    elif args.loss == "BCE":
        criterion = bce_loss
    elif args.loss == "L1":
        criterion = L1_loss
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    # Train the model
    print("Beginning training ...")
    if args.gpu and torch.cuda.is_available(): 
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")
      
    start = time.time()

    train_losses = []
    val_losses = []
    valid_accs = []
    for epoch in range(args.epochs):
        # Train the Model
        model.train() # Change model to 'train' mode
        model.double()
        
        losses = []

        # Forward + Backward + Optimize
        for i_batch, sample_batched in enumerate(trainloader):
            optimizer.zero_grad()

            left = sample_batched['left']
            right = sample_batched['right']
            disp_left = sample_batched['disp_left']
            disp_right = sample_batched['disp_right']
            semantic = sample_batched['semantic']

            model_disp, model_semantic = model(left)
            
            loss = criterion(model_disp, model_semantic, disp_left, disp_right, semantic)

            # print(loss)
            # break

            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())
        else:
            avg_loss = np.mean(losses)
            train_losses.append(avg_loss)
            time_elapsed = time.time() - start
            
            model.eval()
            val_loss = validation(model, criterion, args.test_data, device)
            # val_loss = validation(model, criterion, testloader, device)
            # val_loss = 0
            
            val_losses.append(val_loss)
            print('Epoch [%d/%d], Loss: %.4f, Val Loss: %.4f,  Time (s): %d' % (
                epoch+1, args.epochs, avg_loss, val_loss, time_elapsed))
            
            
    # Plot training curve
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, "b-", label="Training")
    plt.plot(val_losses, "r-", label="Validation")
    plt.legend()
    plt.title("Performance")
    plt.xlabel("Epochs")
    return model

# Arguments for training
args = AttrDict()
args_dict = {
              'gpu':True, 
              'loss':"DEPTH-SEG", 
              'model':"UNet",
              'kernel':3,
              'num_filters':16, 
              'learn_rate':0.0003, 
              'batch_size':4, 
              'epochs':10, 
              'seed':0,
              'downsize_input':True,
              'train_data':depth_train_full,
              'test_data':depth_train_full,
              'num_sem_classes':12,
}
args.update(args_dict)
model_depth_seg = train(args)

"""## Custom training loop

Trains a model for several epochs at a time
"""

# # Debug training loop - Trains a model for several epochs at a time

# def debug_train(model, device, criterion, optimizer, epochs, prev_epochs, graph):
    
#     # Train the model
#     print("Beginning training ...")

#     start = time.time()
#     train_losses = []
#     val_losses = []
#     valid_accs = []
#     for epoch in range(epochs):
#         # Train the Model
#         model.train() # Change model to 'train' mode
#         model.double()
        
#         losses = []

#         # Forward + Backward + Optimize
#         for i_batch, sample_batched in enumerate(trainloader):
#             optimizer.zero_grad()

#             left = sample_batched['left']
#             right = sample_batched['right']
#             disp_left = sample_batched['disp_left']
#             disp_right = sample_batched['disp_right']
#             semantic = sample_batched['semantic']

#             model_disp, model_semantic = model(left)
            
#             loss = criterion(model_disp, model_semantic, disp_left, disp_right, semantic)
#             loss.backward()
#             optimizer.step()

#             losses.append(loss.data.item())
        
            
#         avg_loss = np.mean(losses)
#         model.eval()
#         # val_loss = validation(model, criterion, testloader, device)
#         val_loss = 0

#         train_losses.append(avg_loss)
#         val_losses.append(val_loss)

#         print('Epoch [%d/%d], Loss: %.4f, Val Loss: %.4f, Time: %ds' % (
#               epoch+1+prev_epochs, epochs+prev_epochs, avg_loss, val_loss, time.time() - start))

#     # Plot training curve
#     plt.figure(figsize=(10, 8))
#     plt.plot(train_losses, "b-", label="Training")
#     plt.plot(val_losses, "r-", label="Validation")
#     plt.legend()
#     plt.title(graph)
#     plt.xlabel("Epochs")

#     print('Trained Network on [%d] total epochs thus far' % (epochs + prev_epochs))

#     return model

# debug_args = AttrDict()
# args_dict = {
#               'gpu':True,  
#               'kernel':3,
#               'num_filters':16, 
#               'learn_rate':0.0003, 
#               'batch_size':4, 
#               'seed':100,
#               'train_data':depth_train_full,
#               'test_data':depth_train_full,
#               'downsize_input':True,
# }
# debug_args.update(args_dict)

# torch.set_num_threads(5)
# np.random.seed(args_dict['seed'])

# trainloader = torch.utils.data.DataLoader(debug_args.train_data, batch_size=debug_args.batch_size, shuffle=True)
# testloader = torch.utils.data.DataLoader(debug_args.test_data, batch_size=debug_args.batch_size, shuffle=True)
    
# num_in_channels = 1 if not debug_args.downsize_input else 3

# # LOSS
# criterion = depth_loss

# device = torch.device("cuda")

# segnet = SegNet()
# optimizer_tony = torch.optim.Adam(segnet.parameters(), lr=debug_args.learn_rate)
# segnet.cuda();

# segnet = debug_train(segnet, device, total_loss, optimizer_tony, 2, 12, "Depth Loss SegNet")

# unet = UNet(in_channels=num_in_channels, out_channels=1,
#           kernel=debug_args.kernel, num_features=debug_args.num_filters, num_seg_classes=6)
# optimizer_unet = torch.optim.Adam(unet.parameters(), lr=debug_args.learn_rate)
# unet.cuda();

# unet = debug_train(unet, device, depth_loss, optimizer_unet, 10, 0, "Depth Loss Modified UNet")

"""#Evaluation

## Display
"""

# Display a model's outputs against a dataset
def display(i, data, network):
    sample = data[i]
    device = torch.device("cuda")
    plt.subplot(2, 3, 1), plt.imshow(cpu_img(sample['left'], 3))
    
    item = torch.unsqueeze(sample['left'], 0)

    model_disp, model_semantic = network(item)
    model_disp_detached = model_disp.cpu().detach().numpy()
    model_sem_detached = model_semantic.cpu().detach().numpy()

    disp_result = model_disp_detached[0, 0]
    sem_result = model_sem_detached[0, 0]

    plt.subplot(2, 3, 2), plt.imshow(cpu_img(sample['disp_left']), cmap='jet')
    plt.subplot(2, 3, 3), plt.imshow(disp_result, cmap='jet')
    plt.subplot(2, 3, 4), plt.imshow(cpu_img(sample['semantic']))
    plt.subplot(2, 3, 5), plt.imshow(sem_result)

# plt.figure(figsize=(28, 8))
# display(1, depth_train_full, segnet)

# plt.figure(figsize=(28, 8))
# display(1, depth_train_full, unet)

# plt.figure(figsize=(28, 8))
# display(199, depth_train_full, model_depth)

plt.figure(figsize=(28, 8))
display(199, depth_train_full, model_depth_seg)

"""## Unit tests"""

# # Test model is updating parameters

# model = UNet_Double(in_channels=3, out_channels=1,
#           kernel=3, num_filters=16, num_seg_classes=6)

# criterion = bce
# optimizer = torch.optim.Adam(segnet.parameters(), 0.0001)


# left = sample_batched['left'].unsqueeze(0)
# right = sample_batched['right'].unsqueeze(0)
# disp_left = sample_batched['disp_left'].unsqueeze(0)
# disp_right = sample_batched['disp_right'].unsqueeze(0)
# semantic = sample_batched['semantic'].unsqueeze(0)

# a = list(model.parameters())[0].clone()
# grads = list(model.parameters())[0].grad

# print("params ", a)
# print("grad", grads)


# model_disp, model_semantic = model(left)
# loss = criterion(model_disp, model_semantic, disp_left, disp_right, semantic)
# loss.backward()
# optimizer.step()


# b = list(model.parameters())[0].clone()
# print("params b", b)
# print(torch.equal(a.data, b.data))

# torch.save(unet, 'unet_base_1.pt')
