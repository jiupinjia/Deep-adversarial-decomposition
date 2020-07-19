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


parser = argparse.ArgumentParser(description='EVAL_DERAIN')
parser.add_argument('--dataset', type=str, default='rain100h', metavar='str',
                    help='dataset name from [rain100h, rain100l, rain800, rain800-real, '
                         'did-mdn-test1, did-mdn-test2, rain1400],'
                         '(default: rain100h)')
parser.add_argument('--in_size', type=int, default=512, metavar='N',
                    help='size of input image during eval')
parser.add_argument('--ckptdir', type=str, default='./checkpoints',
                    help='checkpoints dir (default: ./checkpoints)')
parser.add_argument('--net_G', type=str, default='unet_512', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_512)')
parser.add_argument('--save_output', action='store_true', default=False,
                    help='to save the output images')
parser.add_argument('--output_dir', type=str, default='./eval_output', metavar='str',
                    help='evaluation output dir (default: ./eval_output)')
args = parser.parse_args()




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

    if args.dataset == 'rain100h':
        datadir = r'./datasets/Rain100H/val'
        val_dirs = glob.glob(os.path.join(datadir, 'norain-*.png'))
    elif args.dataset == 'rain100l':
        datadir = r'./datasets/Rain100L/val'
        val_dirs = glob.glob(os.path.join(datadir, '*x2.png'))
    elif args.dataset == 'rain800':
        datadir = r'./datasets/Rain800/val'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))
    elif args.dataset == 'rain800-real':
        datadir = r'./datasets/Rain800/test_nature'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))
    elif args.dataset == 'did-mdn-test1':
        datadir = r'./datasets/DID-MDN/val'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))
    elif args.dataset == 'did-mdn-test2':
        datadir = r'./datasets/DID-MDN/testing_fu'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))
    elif args.dataset == 'rain1400':
        datadir = r'./datasets/Rain1400/val/rainy_image'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))

    for idx in range(len(val_dirs)):

        this_dir = val_dirs[idx]

        if args.dataset == 'rain100h':
            gt = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            img_mix = cv2.imread(val_dirs[idx].replace('norain', 'rain'), cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
        elif args.dataset == 'rain100l':
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(val_dirs[idx].replace('x2.png', '.png'), cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        elif args.dataset == 'rain800':
            img = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            gt = img[:, 0:int(w / 2), :]
            img_mix = img[:, int(w / 2):, :]
        elif args.dataset == 'rain800-real':
            img = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            gt = img[:, 0:int(w / 2), :]
            img_mix = img[:, int(w / 2):, :]
        elif args.dataset == 'did-mdn-test1':
            img = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            img_mix = img[:, 0:int(w/2), :]
            gt = img[:, int(w/2):, :]
        elif args.dataset == 'did-mdn-test2':
            img = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            gt = img[:, 0:int(w/2), :]
            img_mix = img[:, int(w/2):, :]
        elif args.dataset == 'rain1400':
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            suff = '_' + this_dir.split('_')[-1]
            this_gt_dir = this_dir.replace('rainy_image', 'ground_truth').replace(suff, '.jpg')
            gt = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        # we recommend to use TF.resize since it was also used during trainig
        # You may also try cv2.resize, but it will produce slightly different results
        img_mix = TF.resize(TF.to_pil_image(img_mix), [args.in_size, args.in_size])
        img_mix = TF.to_tensor(img_mix).unsqueeze(0)

        gt = TF.resize(TF.to_pil_image(gt), [args.in_size, args.in_size])
        gt = TF.to_tensor(gt).unsqueeze(0)

        with torch.no_grad():
            G_pred1 = net_G(img_mix.to(device))[:, 0:3, :, :]
            G_pred2 = net_G(img_mix.to(device))[:, 3:6, :, :]

        G_pred1 = np.array(G_pred1.cpu().detach())
        G_pred1 = G_pred1[0, :].transpose([1, 2, 0])
        G_pred2 = np.array(G_pred2.cpu().detach())
        G_pred2 = G_pred2[0, :].transpose([1, 2, 0])
        gt = np.array(gt.cpu().detach())
        gt = gt[0, :].transpose([1, 2, 0])
        img_mix = np.array(img_mix.cpu().detach())
        img_mix = img_mix[0, :].transpose([1, 2, 0])

        G_pred1[G_pred1 > 1] = 1
        G_pred1[G_pred1 < 0] = 0
        G_pred2[G_pred2 > 1] = 1
        G_pred2[G_pred2 < 0] = 0

        psnr = utils.cpt_rgb_psnr(G_pred1, gt, PIXEL_MAX=1.0)
        ssim = utils.cpt_rgb_ssim(G_pred1, gt)
        running_psnr.append(psnr)
        running_ssim.append(ssim)

        if args.save_output:
            fname = this_dir.split('\\')[-1]
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_input.png'), img_mix)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_gt1.png'), gt)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_output1.png'), G_pred1)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_output2.png'), G_pred2)

        print('id: %d, running psnr: %.4f, running ssim: %.4f'
              % (idx, np.mean(running_psnr), np.mean(running_ssim)))

    print('Dataset: %s, average psnr: %.4f, average ssim: %.4f'
          % (args.dataset, np.mean(running_psnr), np.mean(running_ssim)))



if __name__ == '__main__':

    # args.dataset = 'rain100h'
    # args.dataset = 'rain100l'
    # args.dataset = 'rain800'
    # args.dataset = 'rain800-real'
    # args.dataset = 'did-mdn-test1'
    # args.dataset = 'did-mdn-test2'

    # args.net_G = 'unet_512'
    # args.ckptdir = 'checkpoints'
    # args.save_output = True

    net_G = load_model(args)
    run_eval(args)





