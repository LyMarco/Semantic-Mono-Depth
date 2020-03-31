import torch.nn.functional as F
import os
import torch
import time
import random
from skimage import io, transform, color, data, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pytorch_ssim
import cv2

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
        print("root dir", root_dir)

    def __len__(self):
      return len(self.left)
      # return 1
      
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = {'left': self.left[idx], 'right': self.right[idx], 
                'disp_left': self.disp_left[idx], 'disp_right': self.disp_right[idx],
                'semantic': self.semantic[idx]}
        if self.transform:
            item = self.transform(item)
            
            
        return item

# Data Transformations
# Rescale object repurposed from Pytorch tutorial on Datasets by Sasank Chilamkurthy


class RescaleDispSeg(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        left, right, disp_l, disp_r, semantic = sample['left'], sample[
            'right'], sample['disp_left'], sample['disp_right'], sample['semantic']

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

        return {'left': left, 'right': right, 'disp_left': disp_l, 'disp_right': disp_r, 'semantic': semantic}


data_path = "./images/"

#Original scale is 1242x375
scale = RescaleDispSeg((256, 512))

train_dir = './images/training'
# test_dir = './images/testing'

depth_train_full = DispSegDataset(train_dir, scale)


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
            nn.Conv2d(num_features, num_features *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU())
        self.encode3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
            nn.Conv2d(num_features * 4, num_features *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
        )
        self.encode4 = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(),
            nn.Conv2d(num_features * 8, num_features *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(),
        )

        self.encode5 = nn.Sequential(
            nn.Conv2d(num_features * 8, num_features *
                      16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*16),
            nn.ReLU(),
            nn.Conv2d(num_features * 16, num_features *
                      16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*16),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 16,
                               num_features * 8, kernel_size=2, stride=2)
        )

        self.decode1D = nn.Sequential(
            nn.Conv2d(num_features * 16, num_features *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.Conv2d(num_features * 8, num_features *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                num_features * 8, num_features * 4, kernel_size=2, stride=2)
        )
        self.decode2D = nn.Sequential(
            nn.Conv2d(num_features * 8, num_features *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.Conv2d(num_features * 4, num_features *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                num_features * 4, num_features * 2, kernel_size=2, stride=2)
        )
        self.decode3D = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 2, num_features,
                               kernel_size=2, stride=2)
        )
        self.decode4D = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features,
                      kernel_size=3, padding=1),
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
            nn.Conv2d(num_features * 16, num_features *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.Conv2d(num_features * 8, num_features *
                      8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                num_features * 8, num_features * 4, kernel_size=2, stride=2)
        )
        self.decode2S = nn.Sequential(
            nn.Conv2d(num_features * 8, num_features *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.Conv2d(num_features * 4, num_features *
                      4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                num_features * 4, num_features * 2, kernel_size=2, stride=2)
        )
        self.decode3S = nn.Sequential(
            nn.Conv2d(num_features * 4, num_features *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.Conv2d(num_features * 2, num_features *
                      2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.ReLU(),
            nn.ConvTranspose2d(num_features * 2, num_features,
                               kernel_size=2, stride=2)
        )
        self.decode4S = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
        )
        self.seg = nn.Conv2d(num_features, num_seg_classes,
                             kernel_size=kernel, padding=1)


    def forward(self, x):
        # print("HOLLy", x)
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
        decode1 = self.decode1D(decode1_input)

        decode2_input = torch.cat([decode1, encode3], dim=1)
        decode2 = self.decode2D(decode2_input)

        decode3_input = torch.cat([decode2, encode2], dim=1)
        decode3 = self.decode3D(decode3_input)

        decode4_input = torch.cat([decode3, encode1], dim=1)
        decode4 = self.decode4D(decode4_input)
        self.out_disp = self.disp(decode4)

        # Decoder semantic
        decode1_input = torch.cat([encode4, encode5_out], dim=1)
        decode1 = self.decode1S(decode1_input)

        decode2_input = torch.cat([decode1, encode3], dim=1)
        decode2 = self.decode2S(decode2_input)

        decode3_input = torch.cat([decode2, encode2], dim=1)
        decode3 = self.decode3S(decode3_input)

        decode4_input = torch.cat([decode3, encode1], dim=1)
        decode4 = self.decode4S(decode4_input)
        # Both towers
        self.out_seg = self.seg(decode4)

        return self.out_disp, self.out_seg


# Loss Functions

a_d = 1.0
a_s = 0.1
gamma = 0.85


def SSIM_loss(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    SSIM_n = (2. * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2. + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1. - SSIM) / 2, 0, 1)


def depth_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    # SSIM = pytorch_ssim.SSIM(window_size=11)
    L1 = torch.nn.L1Loss(size_average=False, reduce=False)

    sim_left = torch.mean(1.0 - SSIM_loss(out_disp, target_disp_l)) / 2.0
    sim_right = torch.mean(1.0 - SSIM_loss(out_disp, target_disp_r)) / 2.0

    pixel_loss_l = torch.mean(L1(out_disp, target_disp_l))
    pixel_loss_r = torch.mean(L1(out_disp, target_disp_r))
    # print(pixel_loss.shape)

    left_loss = gamma * sim_left + (1. - gamma) * pixel_loss_l
    right_loss = gamma * sim_right + (1. - gamma) * pixel_loss_r
    # print("loss", left_loss, right_loss)
    # print("L1", pixel_loss_l)
    # print("SIM", sim_left)
    return left_loss + right_loss


def sem_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    CE = nn.CrossEntropyLoss()
    return CE(out_sem, target_sem)


def total_loss(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    d_ap = depth_loss(out_disp, out_sem, target_disp_l,
                      target_disp_r, target_sem)
    d_sem = sem_loss(out_disp, out_sem, target_disp_l,
                     target_disp_r, target_sem)
    return a_d * d_ap + a_s * d_sem


def bce(out_disp, out_sem, target_disp_l, target_disp_r, target_sem):
    BCE = nn.BCELoss()
    return BCE(out_disp, target_disp_l)


def validation(model, criterion, testdata, device):
    losses = []

    for i_batch, test_sample in enumerate(testdata):
        images = test_sample['image']
        masks = test_sample['map']

        reshaped_images = np.transpose(images, (0, 3, 1, 2))
        reshaped_masks = np.reshape(masks, (-1, 1, 256, 512))

        reshaped_images = reshaped_images.to(device)
        reshaped_masks = reshaped_masks.to(device)

        output = model(reshaped_images)

        loss = criterion(output, reshaped_masks)
        losses.append(loss.data.item())

    val_loss = np.mean(losses)
    return val_loss

# Main training loop


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def train(args):
    torch.set_num_threads(5)

    # np.random.seed(args.seed)

    # DATALOADERS
    print("args", args)
    trainloader = torch.utils.data.DataLoader(
        depth_train_full, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        args.test_data, batch_size=args.batch_size, shuffle=True)

    # MODEL
    num_in_channels = 1 if not args.downsize_input else 3
    model = UNet(in_channels=num_in_channels, out_channels=1,
                 kernel=args.kernel, num_features=args.num_filters, num_seg_classes=6)
    # model = UNet_Double(in_channels=num_in_channels, out_channels=1,
    #           kernel=args.kernel, num_filters=args.num_filters, num_seg_classes=6)

    # LOSS FUNCTION
    if args.loss == "DEPTH-SEG":
        criterion = total_loss
    elif args.loss == "DEPTH":
        criterion = depth_loss
    elif args.loss == "SEG":
        criterion = sem_loss
    elif args.loss == "BCE":
        criterion = bce
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

    model.train()  # Change model to 'train' mode
    model.double()

    for epoch in range(args.epochs):
        # Train the Model
        losses = []

        # Forward + Backward + Optimize
        for i_batch, sample_batched in enumerate(trainloader):
            optimizer.zero_grad()

            left = sample_batched['left']
            right = sample_batched['right']
            disp_left = sample_batched['disp_left']
            disp_right = sample_batched['disp_right']
            semantic = sample_batched['semantic']

            # Reshape NHWC
            reshaped_left = np.transpose(left, (0, 3, 1, 2))
            reshaped_right = np.transpose(right, (0, 3, 1, 2))
            # Not sure about these
            reshaped_disp_left = np.reshape(disp_left, (-1, 1, 256, 512))
            reshaped_disp_right = np.reshape(disp_right, (-1, 1, 256, 512))
            reshaped_sem = np.reshape(semantic, (-1, 1, 256, 512))

            reshaped_left = reshaped_left.to(device)
            reshaped_right = reshaped_right.to(device)
            reshaped_disp_left = reshaped_disp_left.to(device)
            reshaped_disp_right = reshaped_disp_right.to(device)
            reshaped_sem = reshaped_sem.to(device)

            # print(reshaped_left)
            model_disp, model_semantic = model(reshaped_left)
            # print(model_output.shape)
            loss = criterion(model_disp, model_semantic,
                             reshaped_disp_left, reshaped_disp_right, reshaped_sem)
            # print("loss", loss, sample_batched)
            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())
        else:
            avg_loss = np.mean(losses)
            train_losses.append(avg_loss)
            time_elapsed = time.time() - start

            model.eval()
            # val_loss = validation(model, criterion, testloader, device)
            val_loss = 0

            val_losses.append(val_loss)
            torch.save(model.state_dict(), 'Epoch_%d.h5' % (epoch+1))
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
    'gpu': True,
    'loss': "DEPTH",
    'kernel': 3,
    'num_filters': 32,
    'learn_rate': 0.001,
    'batch_size': 4,
    'epochs': 40,
    'seed': 0,
    'downsize_input': True,
    'train_data': depth_train_full,
    'test_data': depth_train_full,
}
args.update(args_dict)
unet_depth = train(args)
