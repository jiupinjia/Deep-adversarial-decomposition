import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import random
from skimage.measure import compare_ssim as sk_cpt_ssim
import scipy.stats as st
import skimage.color as skcolor

import torch
torch.cuda.current_device()
import torch.nn as nn
import torchvision.models as torchmodels
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from torchvision.datasets import MNIST, CIFAR10, LSUN, ImageFolder


ALPHA_MAX = 0.6
ALPHA_MIN = 0.4


class DataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_color_jittering=False,
            crop_ratio=(0.9, 1.1)
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_color_jittering = with_color_jittering
        self.crop_ratio = crop_ratio

    def transform(self, img):

        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img = TF.hflip(img)

        if self.with_random_vflip and random.random() > 0.5:
            img = TF.vflip(img)

        if self.with_random_rot90 and random.random() > 0.5:
            img = TF.rotate(img, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = TF.rotate(img, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = TF.rotate(img, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = TF.adjust_hue(img, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img = TF.adjust_saturation(img, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            img = TF.resized_crop(
                img, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img = TF.to_tensor(img)
        return img


    def transform_triplets(self, img, gt1, gt2):

        # resize image and covert to tensor
        img = TF.to_pil_image(img)
        img = TF.resize(img, [self.img_size, self.img_size])

        gt1 = TF.to_pil_image(gt1)
        gt1 = TF.resize(gt1, [self.img_size, self.img_size])

        gt2 = TF.to_pil_image(gt2)
        gt2 = TF.resize(gt2, [self.img_size, self.img_size])

        if self.with_random_hflip and random.random() > 0.5:
            img = TF.hflip(img)
            gt1 = TF.hflip(gt1)
            gt2 = TF.hflip(gt2)

        if self.with_random_vflip and random.random() > 0.5:
            img = TF.vflip(img)
            gt1 = TF.vflip(gt1)
            gt2 = TF.vflip(gt2)

        if self.with_random_rot90 and random.random() > 0.5:
            img = TF.rotate(img, 90)
            gt1 = TF.rotate(gt1, 90)
            gt2 = TF.rotate(gt2, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = TF.rotate(img, 180)
            gt1 = TF.rotate(gt1, 180)
            gt2 = TF.rotate(gt2, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = TF.rotate(img, 270)
            gt1 = TF.rotate(gt1, 270)
            gt2 = TF.rotate(gt2, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = TF.adjust_hue(img, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img = TF.adjust_saturation(img, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6
            gt1 = TF.adjust_hue(gt1, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            gt1 = TF.adjust_saturation(gt1, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6
            gt2 = TF.adjust_hue(gt2, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            gt2 = TF.adjust_saturation(gt2, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            img = TF.resized_crop(
                img, i, j, h, w, size=(self.img_size, self.img_size))
            gt1 = TF.resized_crop(
                gt1, i, j, h, w, size=(self.img_size, self.img_size))
            gt2 = TF.resized_crop(
                gt2, i, j, h, w, size=(self.img_size, self.img_size))

        # to tensor
        img = TF.to_tensor(img)
        gt1 = TF.to_tensor(gt1)
        gt2 = TF.to_tensor(gt2)

        return img, gt1, gt2


def mixture(img1, img2, mix_mode='linear'):
    if mix_mode == 'linear':
        # alpha ~ uniform(ALPHA_MIN, ALPHA_MAX)
        alpha1 = random.random() * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN
        alpha2 = 1 - alpha1
        gt1 = img1 * alpha1
        gt2 = img2 * alpha2
        img_mix = gt1 + gt2
    else:
        alpha1 = 1.0
        alpha2 = 1.0
        gt1 = img1 * alpha1
        gt2 = img2 * alpha2
        img_mix = gt1 + gt2 + 0.1*torch.randn_like(gt1)
        img_mix = np.clip(img_mix, a_max=1.0, a_min=0.0)


    data = {
        'input': img_mix,
        'gt1': gt1,
        'gt2': gt2,
        'alpha1': alpha1,
        'alpha2': alpha2
    }
    return data



class TwoFoldersUnmixDataset(Dataset):

    def __init__(self, root_dir_1, root_dir_2, img_size, mix_mode='linear', suff='.jpg', is_train=True):
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.dirs_1 = glob.glob(os.path.join(self.root_dir_1, '*'+suff))
        self.dirs_2 = glob.glob(os.path.join(self.root_dir_2, '*'+suff))
        self.img_size = img_size
        self.mix_mode = mix_mode
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return max(len(self.dirs_1), len(self.dirs_2))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input: color image (h x w x 3)
        this_dir = self.dirs_1[idx]
        img1 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = self.augm.transform(img1)

        rand_idx = random.randint(0, len(self.dirs_2)-1)
        this_dir = self.dirs_2[rand_idx]
        img2 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = self.augm.transform(img2)

        data = mixture(img1, img2, self.mix_mode)

        return data



class Cifar10UnmixDataset(CIFAR10):

    def __init__(self, root_dir, img_size, is_train=True):
        super(Cifar10UnmixDataset, self).__init__(
            root=root_dir, train=is_train, download=True)
        self.img_size = img_size
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1 = self.data[idx]
        img1 = self.augm.transform(img1)

        rand_idx = random.randint(0, len(self.data)-1)
        img2 = self.data[rand_idx]
        img2 = self.augm.transform(img2)

        data = mixture(img1, img2)

        return data



class MNISTUnmixDataset(MNIST):

    def __init__(self, root_dir, img_size, is_train=True):
        super(MNISTUnmixDataset, self).__init__(
            root=root_dir, train=is_train, download=True)
        self.img_size = img_size
        self.augm = DataAugmentation(
            img_size=self.img_size,
            with_random_hflip=False,
            with_random_crop=False)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1 = self.data[idx]
        img1 = np.stack([img1, img1, img1], axis=2)
        img1 = self.augm.transform(img1)

        rand_idx = random.randint(0, len(self.data)-1)
        img2 = self.data[rand_idx]
        img2 = np.stack([img2, img2, img2], axis=2)
        img2 = self.augm.transform(img2)

        data = mixture(img1, img2)

        return data




class ImageNetUnmixDataset(Dataset):

    def __init__(self, root_dir, img_size, is_train=True):
        self.root_dir = root_dir
        self.img_dirs = self.parse_dirs(is_train)
        self.img_size = img_size

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def parse_dirs(self, is_train):

        if is_train:
            # parse the training folders
            img_dirs = list(glob.iglob(os.path.join(self.root_dir, '*/*.JPEG')))
            return img_dirs
        else:
            # parse the val folders
            img_dirs = glob.glob(os.path.join(self.root_dir, '*.JPEG'))
            return img_dirs

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input: color image (h x w x 3)
        this_dir = self.img_dirs[idx]
        img1 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = self.augm.transform(img1)

        rand_idx = random.randint(0, len(self.img_dirs)-1)
        this_dir = self.img_dirs[rand_idx]
        img2 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = self.augm.transform(img2)

        data = mixture(img1, img2)

        return data



class Rain100HDataset(Dataset):

    def __init__(self, root_dir, img_size, suff='.png', is_train=True):
        self.root_dir = root_dir
        self.dirs = glob.glob(os.path.join(self.root_dir, 'norain-*'+suff))
        self.img_size = img_size
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input: color image (h x w x 3)
        this_dir = self.dirs[idx]
        gt1 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        this_dir = self.dirs[idx].replace('norain', 'rain')
        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        this_dir = self.dirs[idx].replace('norain', 'rainstreak')
        gt2 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        gt2 = cv2.cvtColor(gt2, cv2.COLOR_BGR2RGB)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




class Rain100LDataset(Dataset):

    def __init__(self, root_dir, img_size, suff='.png', is_train=True):
        self.root_dir = root_dir
        self.dirs = glob.glob(os.path.join(self.root_dir, '*x2'+suff))
        self.img_size = img_size
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input: color image (h x w x 3)
        this_dir = self.dirs[idx]
        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        this_dir = self.dirs[idx].replace('x2.png', '.png')
        gt1 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        gt2 = np.zeros_like(gt1)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data



class Rain800Dataset(Dataset):

    def __init__(self, root_dir, img_size, suff='.png', gt_left=True, is_train=True):
        self.root_dir = root_dir
        self.dirs = glob.glob(os.path.join(self.root_dir, '*'+suff))
        self.img_size = img_size
        self.gt_left = gt_left
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input: color image (h x w x 3)
        this_dir = self.dirs[idx]
        img = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        if self.gt_left:
            gt1 = img[:, 0:int(w/2), :]
            img_mix = img[:, int(w/2):, :]
        else:
            img_mix = img[:, 0:int(w / 2), :]
            gt1 = img[:, int(w / 2):, :]
        gt2 = np.zeros_like(gt1)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




class Rain1400Dataset(Dataset):

    def __init__(self, root_dir, img_size=512, is_train=True):
        self.root_dir = root_dir
        self.dirs = glob.glob(os.path.join(self.root_dir, 'rainy_image', '*.jpg'))
        self.img_size = img_size
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_dir = self.dirs[idx]
        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        suff = '_' + this_dir.split('_')[-1]
        this_gt_dir = this_dir.replace('rainy_image', 'ground_truth').replace(suff, '.jpg')
        gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        gt2 = np.zeros_like(gt1)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




class BDNDataset(Dataset):

    def __init__(self, root_dir, img_size=512, is_train=True):
        self.root_dir = root_dir
        self.dirs = glob.glob(os.path.join(self.root_dir, 'I', '*.jpg'))
        self.img_size = img_size
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True,
                crop_ratio=(1.0, 1.0))
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_dir = self.dirs[idx]
        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        this_gt_dir = this_dir.replace('I', 'B')
        gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        gt2 = np.zeros_like(gt1)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




class Syn3Dataset(Dataset):

    def __init__(self, root_dir, subset, img_size=512, is_train=True):
        self.root_dir = root_dir
        defocused_dirs = glob.glob(os.path.join(self.root_dir, 'defocused/C', '*.png'))
        focused_dirs = glob.glob(os.path.join(self.root_dir, 'focused/C', '*.png'))
        ghosting_dirs = glob.glob(os.path.join(self.root_dir, 'ghosting/C', '*.png'))
        if subset == 'defocused':
            self.dirs = defocused_dirs
        elif subset == 'focused':
            self.dirs = focused_dirs
        elif subset == 'ghosting':
            self.dirs = ghosting_dirs
        elif subset == 'all':
            self.dirs = defocused_dirs + focused_dirs + ghosting_dirs

        self.img_size = img_size
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True,
                crop_ratio=(1.0, 1.0))
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_dir = self.dirs[idx]
        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        this_gt_dir = this_dir.replace('/C', '/B')
        gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        gt2 = np.zeros_like(gt1)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




class XZhangDataset(Dataset):

    def __init__(self, root_dir, img_size=512, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train
        # create a vignetting mask
        g_mask = self._gkern(800, 3)
        self.g_mask = np.dstack((g_mask, g_mask, g_mask))

        if is_train:
            syn_trans_dirs = glob.glob(os.path.join(self.root_dir, 'syn/transmission_layer', '*.jpg'))
            real_trans_dirs = glob.glob(os.path.join(self.root_dir, 'real/transmission_layer', '*.jpg'))
            self.trans_dirs = syn_trans_dirs + real_trans_dirs*100
            self.syn_refl_dirs = glob.glob(os.path.join(self.root_dir, 'syn/reflection_layer', '*.jpg'))
        else:
            self.trans_dirs = glob.glob(os.path.join(self.root_dir, 'transmission_layer', '*.jpg'))

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True,
                crop_ratio=(0.5, 2.0))
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.trans_dirs)

    # functions for synthesizing images with reflection (details in the paper)
    def _gkern(self, kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel / kernel.max()
        return kernel

    def _syn_data(self, t, r):
        sigma = random.random() * 7 + 1
        t = np.power(t, 2.2)
        r = np.power(r, 2.2)

        sz = int(2 * np.ceil(2 * sigma) + 1)
        r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
        blend = r_blur + t

        att = 1.08 + np.random.random() / 10.0

        for i in range(3):
            maski = blend[:, :, i] > 1
            mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
            r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
        r_blur[r_blur >= 1] = 1
        r_blur[r_blur <= 0] = 0

        h, w = r_blur.shape[0:2]
        neww = np.random.randint(0, 800 - w - 1)
        newh = np.random.randint(0, 800 - h - 1)
        alpha1 = self.g_mask[newh:newh + h, neww:neww + w, :]
        alpha2 = 1 - np.random.random() / 5.0
        r_blur_mask = np.multiply(r_blur, alpha1)
        blend = r_blur_mask + t * alpha2

        t = np.power(t, 1 / 2.2)
        r_blur_mask = np.power(r_blur_mask, 1 / 2.2)
        blend = np.power(blend, 1 / 2.2)
        blend[blend >= 1] = 1
        blend[blend <= 0] = 0

        return t, r_blur_mask, blend

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_trans_dir = self.trans_dirs[idx]

        if self.is_train and 'syn' in this_trans_dir:
            # train with syn data
            img_t = cv2.imread(this_trans_dir, cv2.IMREAD_COLOR)
            img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
            this_refl_dir = self.syn_refl_dirs[random.randint(0, len(self.syn_refl_dirs) - 1)]
            img_r = cv2.imread(this_refl_dir, cv2.IMREAD_COLOR)
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

            img_t = cv2.resize(img_t, (self.img_size, self.img_size))/255.
            img_r = cv2.resize(img_r, (self.img_size, self.img_size))/255.
            t, _, mix = self._syn_data(img_t, img_r)
            gt1 = np.uint8(t*255)
            img_mix = np.uint8(mix*255)
        else:
            # train and eval with real data
            gt1 = cv2.imread(this_trans_dir, cv2.IMREAD_COLOR)
            gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)
            this_dir = this_trans_dir.replace('transmission_layer', 'blended')
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        gt2 = np.zeros_like(gt1)
        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




class ISTDDataset(Dataset):

    def __init__(self, root_dir, img_size=512, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        if self.is_train:
            self.dirs = glob.glob(os.path.join(self.root_dir, 'train_A', '*.png'))
        else:
            self.dirs = glob.glob(os.path.join(self.root_dir, 'test_A', '*.png'))

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_random_crop=True,
                with_random_rot180=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_dir = self.dirs[idx]
        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        if self.is_train:
            this_gt_dir = this_dir.replace('train_A', 'train_C')
        else:
            this_gt_dir = this_dir.replace('test_A', 'test_C')
        gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        gt2 = np.zeros_like(gt1)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




class SRDDataset(Dataset):

    def __init__(self, root_dir, img_size=512, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        self.dirs = glob.glob(os.path.join(self.root_dir, 'shadow', '*.jpg'))

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_random_crop=True,
                with_random_rot180=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_dir = self.dirs[idx]
        img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)

        if self.is_train:
            this_gt_dir = this_dir.replace('shadow', 'shadow_free')[:-4] + '_no_shadow.jpg'
        else:
            this_gt_dir = this_dir.replace('shadow', 'shadow_free')[:-4] + '_free.jpg'
        gt1 = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
        gt1 = cv2.cvtColor(gt1, cv2.COLOR_BGR2RGB)

        gt2 = np.zeros_like(gt1)

        img_mix, gt1, gt2 = self.augm.transform_triplets(img_mix, gt1, gt2)

        data = {
            'input': img_mix,
            'gt1': gt1,
            'gt2': gt2
        }

        return data




def get_loaders(args):

    if args.dataset == 'dogs':
        training_set = TwoFoldersUnmixDataset(
            root_dir_1=r'./datasets/dogs/train',
            root_dir_2=r'./datasets/dogs/train',
            img_size=128, suff='.jpg', is_train=True)
        val_set = TwoFoldersUnmixDataset(
            root_dir_1=r'./datasets/dogs/val',
            root_dir_2=r'./datasets/dogs/val',
            img_size=128, suff='.jpg', is_train=False)

    elif args.dataset == 'dogsflowers':
        training_set = TwoFoldersUnmixDataset(
            root_dir_1=r'./datasets/dogs/train',
            root_dir_2=r'./datasets/flowers/train',
            img_size=128, mix_mode=args.mix_mode, suff='.jpg', is_train=True)
        val_set = TwoFoldersUnmixDataset(
            root_dir_1=r'./datasets/dogs/train',
            root_dir_2=r'./datasets/flowers/train',
            img_size=128, mix_mode=args.mix_mode, suff='.jpg', is_train=False)

    elif args.dataset == 'cifar10':
        training_set = Cifar10UnmixDataset(
            root_dir=r'./datasets/cifar10', img_size=64, is_train=True)
        val_set = Cifar10UnmixDataset(
            root_dir=r'./datasets/cifar10', img_size=64, is_train=False)

    elif args.dataset == 'mnist':
        training_set = MNISTUnmixDataset(
            root_dir=r'./datasets/mnist', img_size=64, is_train=True)
        val_set = MNISTUnmixDataset(
            root_dir=r'./datasets/mnist', img_size=64, is_train=False)

    elif args.dataset == 'tinyimagenet':
        training_set = ImageNetUnmixDataset(
            root_dir=r'./datasets/tiny-imagenet-200/train', img_size=64, is_train=True)
        val_set = ImageNetUnmixDataset(
            root_dir=r'./datasets/tiny-imagenet-200/val/images', img_size=64, is_train=False)

    elif args.dataset == 'lsun':
        training_set = TwoFoldersUnmixDataset(
            root_dir_1=r'/scratch/jpye_root/jpye/zzhengxi/lsun-master/raw_data/train/classroom',
            root_dir_2=r'/scratch/jpye_root/jpye/zzhengxi/lsun-master/raw_data/train/church_outdoor',
            img_size=256, suff='.webp', is_train=True)
        val_set = TwoFoldersUnmixDataset(
            root_dir_1=r'/scratch/jpye_root/jpye/zzhengxi/lsun-master/raw_data/val/classroom',
            root_dir_2=r'/scratch/jpye_root/jpye/zzhengxi/lsun-master/raw_data/val/church_outdoor',
            img_size=256, suff='.webp', is_train=False)

    elif args.dataset == 'imagenet':
        training_set = ImageNetUnmixDataset(
            root_dir=r'/scratch/jpye_root/jpye/zzhengxi/ILSVRC2012/ILSVRC2012_img_train',
            img_size=256, is_train=True)
        val_set = ImageNetUnmixDataset(
            root_dir=r'/scratch/jpye_root/jpye/zzhengxi/ILSVRC2012/ILSVRC2012_img_val',
            img_size=256, is_train=False)

    elif args.dataset == 'imagenetsubset':
        training_set = ImageNetUnmixDataset(
            root_dir=r'/scratch/jpye_root/jpye/zzhengxi/ILSVRC2012/ILSVRC2012_img_train',
            img_size=256, is_train=True)
        val_set = ImageNetUnmixDataset(
            root_dir=r'/scratch/jpye_root/jpye/zzhengxi/ILSVRC2012/ILSVRC2012_img_val',
            img_size=256, is_train=False)
        m = len(training_set)
        training_set = Subset(training_set, indices=np.random.choice(m, int(0.1*m)).tolist())
        m = len(val_set)
        val_set = Subset(val_set, indices=np.random.choice(m, int(0.1 * m)).tolist())

    elif args.dataset == 'rain100h':
        training_set = Rain100HDataset(
            root_dir=r'./datasets/Rain100H/train',
            img_size=512, suff='.png', is_train=True)
        val_set = Rain100HDataset(
            root_dir=r'./datasets/Rain100H/val',
            img_size=512, suff='.png', is_train=False)

    elif args.dataset == 'rain100l':
        training_set = Rain100LDataset(
            root_dir=r'./datasets/Rain100L/train',
            img_size=512, suff='.png', is_train=True)
        val_set = Rain100LDataset(
            root_dir=r'./datasets/Rain100L/val',
            img_size=512, suff='.png', is_train=False)

    elif args.dataset == 'rain800':
        training_set = Rain800Dataset(
            root_dir=r'./datasets/Rain800/train',
            img_size=512, suff='.jpg', gt_left=True, is_train=True)
        val_set = Rain800Dataset(
            root_dir=r'./datasets/Rain800/val',
            img_size=512, suff='.jpg', gt_left=True, is_train=False)

    elif args.dataset == 'did-mdn':
        training_set = Rain800Dataset(
            root_dir=r'./datasets/DID-MDN/train',
            img_size=512, suff='.jpg', gt_left=False, is_train=True)
        val_set = Rain800Dataset(
            root_dir=r'./datasets/DID-MDN/val',
            img_size=512, suff='.jpg', gt_left=False, is_train=False)

    elif args.dataset == 'rain1400':
        training_set = Rain1400Dataset(
            root_dir=r'./datasets/Rain1400/train', img_size=512, is_train=True)
        val_set = Rain1400Dataset(
            root_dir=r'./datasets/Rain1400/val', img_size=512, is_train=False)

    elif args.dataset == 'bdn':
        training_set = BDNDataset(
            root_dir=r'./datasets/BDN/ref_data_train', img_size=256, is_train=True)
        val_set = BDNDataset(
            root_dir=r'./datasets/BDN/ref_data_test', img_size=256, is_train=False)

    elif args.dataset == 'syn3-defocused':
        training_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/train', subset='defocused', img_size=512, is_train=True)
        val_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/val', subset='defocused', img_size=512, is_train=False)

    elif args.dataset == 'syn3-focused':
        training_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/train', subset='focused', img_size=512, is_train=True)
        val_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/val', subset='focused', img_size=512, is_train=False)

    elif args.dataset == 'syn3-ghosting':
        training_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/train', subset='ghosting', img_size=512, is_train=True)
        val_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/val', subset='ghosting', img_size=512, is_train=False)

    elif args.dataset == 'syn3-all':
        training_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/train', subset='all', img_size=512, is_train=True)
        val_set = Syn3Dataset(
            root_dir=r'./datasets/Syn3/val', subset='all', img_size=512, is_train=False)

    elif args.dataset == 'xzhang':
        training_set = XZhangDataset(
            root_dir=r'./datasets/XZhang/train', img_size=512, is_train=True)
        val_set = XZhangDataset(
            root_dir=r'./datasets/XZhang/val_real', img_size=512, is_train=False)

    elif args.dataset == 'istd':
        training_set = ISTDDataset(
            root_dir=r'./datasets/ISTD/train', img_size=256, is_train=True)
        val_set = ISTDDataset(
            root_dir=r'./datasets/ISTD/test', img_size=256, is_train=False)

    elif args.dataset == 'srd':
        training_set = SRDDataset(
            root_dir=r'./datasets/SRD/Train', img_size=512, is_train=True)
        val_set = SRDDataset(
            root_dir=r'./datasets/SRD/Test', img_size=512, is_train=False)

    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [dogsflowers, cifar10, mnist, imagenet, '
                         'rain100h, rain100l, rain800, did-mdn, rain1400'
                         'bdn, syn3-defocused, syn3-focused, syn3-ghosting, syn3-all, xzhang,'
                         'istd, srd])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=8)
                   for x in ['train', 'val']}

    return dataloaders




def make_numpy_grid(tensor_data, enhance=False):

    tensor_data = tensor_data.detach()
    if enhance:
        tensor_data = 1.5*tensor_data
    vis = utils.make_grid(tensor_data)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    return vis


def clip_01(x):
    x[x>1.0] = 1.0
    x[x<0] = 0
    return x


def cpt_rgb_psnr(img, img_gt, PIXEL_MAX):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


def cpt_psnr(batch, batch_gt, PIXEL_MAX):
    batch = clip_01(batch)
    batch_gt = clip_01(batch_gt)
    mse = torch.mean((batch - batch_gt) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    psnr = torch.mean(psnr)
    return psnr


def cpt_rgb_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    SSIM = 0
    for i in range(3):
        tmp = img[:, :, i]
        tmp_gt = img_gt[:, :, i]
        ssim = sk_cpt_ssim(tmp, tmp_gt)
        SSIM = SSIM + ssim
    return SSIM / 3.0


def cpt_ssim(batch, batch_gt):
    batch = clip_01(batch)
    batch_gt = clip_01(batch_gt)

    batch = np.array(batch.cpu())
    batch_gt = np.array(batch_gt.cpu())
    SSIM = 0
    m = batch_gt.shape[0]
    for i in range(m):
        img = batch[i,:].transpose([1,2,0])
        gt = batch_gt[i,:].transpose([1,2,0])
        ssim = cpt_rgb_ssim(img, gt)
        SSIM = SSIM + ssim

    return SSIM / m


def cpt_rgb_labrmse(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    img = skcolor.rgb2lab(img)
    img_gt = skcolor.rgb2lab(img_gt)
    rmse = np.mean(np.abs(img - img_gt)) * 3
    return rmse

def cpt_labrmse(batch, batch_gt):
    batch = clip_01(batch).cpu()
    batch_gt = clip_01(batch_gt).cpu()
    batch = np.array(batch.cpu())
    batch_gt = np.array(batch_gt.cpu())
    RMSE = 0
    m = batch_gt.shape[0]
    for i in range(m):
        img = batch[i, :].transpose([1, 2, 0])
        gt = batch_gt[i, :].transpose([1, 2, 0])
        rmse = cpt_rgb_labrmse(img, gt)
        RMSE = RMSE + rmse

    return RMSE / m



def insert_synfake(fake_cat, batch):
    m = int(fake_cat.shape[0]*0.5)
    sub_batch_gt1 = batch['gt1'][0: m, :, :, :]
    sub_batch_gt2 = batch['gt2'][0: m, :, :, :]
    alpha_ = random.random() * (ALPHA_MAX - ALPHA_MIN) + ALPHA_MIN
    mix1 = sub_batch_gt1 * alpha_ + sub_batch_gt2 * (1 - alpha_)
    mix2 = sub_batch_gt1 * (1 - alpha_) + sub_batch_gt2 * alpha_
    syn_cat = torch.cat((mix1, mix2), dim=1)
    fake_cat[0:m, :, :, :] = syn_cat

    return fake_cat



def visulize_ouput(img_in, epoch_id, b_id, inpath):

    img_in = make_numpy_grid(img_in)

    if not os.path.exists(inpath):
        os.mkdir(inpath)

    this_img_name = 'epoch_' + str(epoch_id) + \
                    '_batch_id_' + str(b_id) + '.jpg'
    plt.imsave(os.path.join(inpath, this_img_name), img_in)


def read_and_crop_img(fname, d=32):

    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.

    img_h, img_w, c = img.shape

    new_h = img_h - img_h % d
    new_w = img_w - img_w % d

    y1 = int((img_h - new_h)/2)
    x1 = int((img_w - new_w)/2)
    y2 = int((img_h + new_h)/2)
    x2 = int((img_w + new_w)/2)

    img_cropped = img[y1:y2,x1:x2,:]

    return img_cropped



def read_and_mix_images(fname1, fname2, d=32):

    # read image1
    img1 = cv2.imread(fname1, cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.
    img1_h, img1_w, c = img1.shape

    # read image2
    img2 = cv2.imread(fname2, cv2.IMREAD_COLOR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.
    img2_h, img2_w, c = img2.shape

    # get the minimum h and w
    min_h = min(img1_h, img2_h)
    min_w = min(img1_w, img2_w)

    # get cropped h and w
    new_h = min_h - min_h % d
    new_w = min_w - min_w % d

    # crop image1
    y1 = int((img1_h - new_h) / 2)
    x1 = int((img1_w - new_w) / 2)
    y2 = int((img1_h + new_h) / 2)
    x2 = int((img1_w + new_w) / 2)
    img1_cropped = img1[y1:y2, x1:x2, :]

    # crop image2
    y1 = int((img2_h - new_h) / 2)
    x1 = int((img2_w - new_w) / 2)
    y2 = int((img2_h + new_h) / 2)
    x2 = int((img2_w + new_w) / 2)
    img2_cropped = img2[y1:y2, x1:x2, :]

    # let's mix the cropped images
    alpha = 0.5
    img_mixed = alpha*img1_cropped + (1-alpha)*img2_cropped

    return img1_cropped, img2_cropped, img_mixed


def np2torch_tensor(imgs, device):
    for i in range(len(imgs)):
        imgs[i] = torch.from_numpy(imgs[i].transpose([2, 0, 1])[None, :]).float().to(device)
    return imgs






########################################################
########################################################

"""
The following part of the code is adapted from the project:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""



class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images




def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
