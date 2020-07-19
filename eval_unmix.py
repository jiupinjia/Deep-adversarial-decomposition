import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse

import utils

import torch
import torchvision.transforms.functional as TF
import cyclegan_networks as cycnet

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='EVAL_DEREFLECTION')
parser.add_argument('--dataset', type=str, default='dogs', metavar='str',
                    help='dataset name from [dogs, imagenetsubset],'
                         '(default: dogs)')
parser.add_argument('--in_size', type=int, default=256, metavar='N',
                    help='size of input image during eval')
parser.add_argument('--ckptdir', type=str, default='./checkpoints',
                    help='checkpoints dir (default: ./checkpoints)')
parser.add_argument('--net_G', type=str, default='unet_256', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_512)')
parser.add_argument('--save_output', action='store_true', default=False,
                    help='to save the output images')
parser.add_argument('--output_dir', type=str, default='./eval_output', metavar='str',
                    help='evaluation output dir (default: ./eval_output)')
args = parser.parse_args()


def cpt_psnr_ssim(G_pred1, G_pred2, gt1, gt2):
    psnr1 = 0.5*utils.cpt_rgb_psnr(G_pred1, gt1, PIXEL_MAX=1.0) + \
            0.5*utils.cpt_rgb_psnr(G_pred2, gt2, PIXEL_MAX=1.0)
    psnr2 = 0.5*utils.cpt_rgb_psnr(G_pred1, gt2, PIXEL_MAX=1.0) + \
            0.5*utils.cpt_rgb_psnr(G_pred2, gt1, PIXEL_MAX=1.0)

    ssim1 = 0.5*utils.cpt_rgb_ssim(G_pred1, gt1) + \
            0.5*utils.cpt_rgb_ssim(G_pred2, gt2)
    ssim2 = 0.5*utils.cpt_rgb_ssim(G_pred1, gt2) + \
            0.5*utils.cpt_rgb_ssim(G_pred2, gt1)

    return max(psnr1, psnr2), max(ssim1, ssim2)



def load_model(args):

    net_G = cycnet.define_G(
                input_nc=3, output_nc=6, ngf=64, netG=args.net_G, use_dropout=False, norm='none').to(device)
    print('loading the best checkpoint...')
    checkpoint = torch.load(os.path.join(args.ckptdir, 'best_ckpt.pt'))
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    net_G.to(device)
    net_G.eval()

    return net_G



def run_eval(args):

    print('running evaluation...')

    if args.save_output:
        if os.path.exists(args.output_dir) is False:
            os.mkdir(args.output_dir)

    running_psnr = []
    running_ssim = []

    if args.dataset == 'dogs':
        datadir = r'./datasets/dogs/val'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))
    elif args.dataset == 'dogsflowers':
        datadir = r'./datasets/dogs/val'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))
        datadir_flowers = r'./datasets/flowers/val'
        val_dirs_flowers = glob.glob(os.path.join(datadir_flowers, '*.jpg'))
    elif args.dataset == 'imagenetsubset':
        datadir = r'/scratch/jpye_root/jpye/zzhengxi/ILSVRC2012/ILSVRC2012_img_val'
        val_dirs = glob.glob(os.path.join(datadir, '*.JPEG'))
    elif args.dataset == 'lsun':
        datadir = r'/scratch/jpye_root/jpye/zzhengxi/lsun-master/raw_data/val/classroom'
        val_dirs = glob.glob(os.path.join(datadir, '*.webp'))
        datadir2 = r'/scratch/jpye_root/jpye/zzhengxi/lsun-master/raw_data/val/church_outdoor'
        val_dirs2 = glob.glob(os.path.join(datadir2, '*.webp'))

    for idx in range(len(val_dirs)):

        if args.dataset == 'dogs':
            img1 = cv2.imread(val_dirs[idx], cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(val_dirs[-idx], cv2.IMREAD_COLOR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        elif args.dataset == 'dogsflowers':
            img1 = cv2.imread(val_dirs[idx], cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(val_dirs_flowers[-idx], cv2.IMREAD_COLOR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        elif args.dataset == 'imagenetsubset':
            img1 = cv2.imread(val_dirs[idx], cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(val_dirs[-idx], cv2.IMREAD_COLOR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        elif args.dataset == 'lsun':
            img1 = cv2.imread(val_dirs[idx], cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(val_dirs2[-idx], cv2.IMREAD_COLOR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)



        # we recommend to use TF.resize since it was also used during trainig
        # You may also try cv2.resize, but it will produce slightly different results
        gt1 = TF.resize(TF.to_pil_image(img1), [args.in_size, args.in_size])
        gt1 = 0.5*TF.to_tensor(gt1).unsqueeze(0)
        gt2 = TF.resize(TF.to_pil_image(img2), [args.in_size, args.in_size])
        gt2 = 0.5*TF.to_tensor(gt2).unsqueeze(0)
        img_mix = gt1 + gt2

        with torch.no_grad():
            out = net_G(img_mix.to(device))
            G_pred1 = out[:, 0:3, :, :]
            G_pred2 = out[:, 3:6, :, :]

        G_pred1 = np.array(G_pred1.cpu().detach())
        G_pred1 = G_pred1[0, :].transpose([1, 2, 0])
        G_pred2 = np.array(G_pred2.cpu().detach())
        G_pred2 = G_pred2[0, :].transpose([1, 2, 0])
        gt1 = np.array(gt1.cpu().detach())
        gt1 = gt1[0, :].transpose([1, 2, 0])
        gt2 = np.array(gt2.cpu().detach())
        gt2 = gt2[0, :].transpose([1, 2, 0])
        img_mix = np.array(img_mix.cpu().detach())
        img_mix = img_mix[0, :].transpose([1, 2, 0])

        G_pred1[G_pred1 > 0.5] = 0.5
        G_pred1[G_pred1 < 0] = 0
        G_pred2[G_pred2 > 0.5] = 0.5
        G_pred2[G_pred2 < 0] = 0

        psnr, ssim = cpt_psnr_ssim(G_pred1, G_pred2, gt1, gt2)
        running_psnr.append(psnr)
        running_ssim.append(ssim)

        if args.save_output:
            fname = val_dirs[idx].split('\\')[-1]
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_input.png'), img_mix)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_gt1.png'), gt1*2.0)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_gt2.png'), gt2*2.0)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_output1.png'), G_pred1*2.0)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_output2.png'), G_pred2*2.0)

        print('id: %d, running psnr: %.4f, running ssim: %.4f'
              % (idx, np.mean(running_psnr), np.mean(running_ssim)))

    print('Dataset: %s, average psnr: %.4f, average ssim: %.4f'
          % (args.dataset, np.mean(running_psnr), np.mean(running_ssim)))



if __name__ == '__main__':

    # args.dataset = 'dogsflowers'
    # args.net_G = 'unet_128'
    # args.in_size = 128
    # args.ckptdir = 'checkpoints'

    # args.dataset = 'mnist'
    # args.net_G = 'unet_64'
    # args.in_size = 64
    # args.ckptdir = 'checkpoints'

    args.save_output = True

    net_G = load_model(args)
    run_eval(args)





