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
parser.add_argument('--dataset', type=str, default='xzhang', metavar='str',
                    help='dataset name from [xzhang, ceilnet-syn, bdn, syn3-defocused, syn3-focused, syn3-ghosting, syn3-all],'
                         '(default: xzhang)')
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

    if args.dataset == 'xzhang':
        datadir = r'./datasets/XZhang/val_real/blended'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))
    elif args.dataset == 'syn3-all':
        datadir = r'datasets/Syn3/val'
        defocused_dir = glob.glob(os.path.join(datadir, 'defocused/C', '*.png'))
        focused_dir = glob.glob(os.path.join(datadir, 'focused/C', '*.png'))
        ghosting_dir = glob.glob(os.path.join(datadir, 'ghosting/C', '*.png'))
        val_dirs = defocused_dir + focused_dir + ghosting_dir
    elif args.dataset == 'syn3-defocused':
        datadir = r'datasets/Syn3/val'
        val_dirs = glob.glob(os.path.join(datadir, 'defocused/C', '*.png'))
    elif args.dataset == 'syn3-focused':
        datadir = r'datasets/Syn3/val'
        val_dirs = glob.glob(os.path.join(datadir, 'focused/C', '*.png'))
    elif args.dataset == 'syn3-ghosting':
        datadir = r'datasets/Syn3/val'
        val_dirs = glob.glob(os.path.join(datadir, 'ghosting/C', '*.png'))
    elif args.dataset == 'bdn':
        datadir = r'datasets/BDN/ref_data_test'
        val_dirs = glob.glob(os.path.join(datadir, 'I', '*.jpg'))



    for idx in range(len(val_dirs)):

        this_dir = val_dirs[idx]

        if args.dataset == 'xzhang':
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(val_dirs[idx].replace('blended', 'transmission_layer'), cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            p = 0
            img_mix = np.pad(img_mix, ((p, p), (p, p), (0, 0)), 'constant')
            gt = np.pad(gt, ((p, p), (p, p), (0, 0)), 'constant')
        elif args.dataset in ['syn3-all', 'syn3-defocused', 'syn3-focused', 'syn3-ghosting']:
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            this_gt_dir = this_dir.replace('/C', '/B')
            gt = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        elif args.dataset is 'bdn':
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            this_gt_dir = this_dir.replace('I', 'B')
            gt = cv2.imread(this_gt_dir, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)


        # we recommend to use TF.resize since it was also used during trainig
        # You may also try cv2.resize, but it will produce slightly different results
        img_mix = TF.resize(TF.to_pil_image(img_mix), [args.in_size, args.in_size])
        img_mix = TF.to_tensor(img_mix).unsqueeze(0)

        gt = TF.resize(TF.to_pil_image(gt), [args.in_size, args.in_size])
        gt = TF.to_tensor(gt).unsqueeze(0)

        with torch.no_grad():
            G_pred = net_G(img_mix.to(device))[:, 0:3, :, :]

        G_pred = np.array(G_pred.cpu().detach())
        G_pred = G_pred[0, :].transpose([1, 2, 0])
        gt = np.array(gt.cpu().detach())
        gt = gt[0, :].transpose([1, 2, 0])
        img_mix = np.array(img_mix.cpu().detach())
        img_mix = img_mix[0, :].transpose([1, 2, 0])

        G_pred[G_pred > 1.0] = 1.0
        G_pred[G_pred < 0] = 0

        psnr = utils.cpt_rgb_psnr(G_pred, gt, PIXEL_MAX=1.0)
        ssim = utils.cpt_rgb_ssim(G_pred, gt)
        running_psnr.append(psnr)
        running_ssim.append(ssim)

        if args.save_output:
            fname = this_dir.split('\\')[-1]
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_input.png'), img_mix)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_gt.png'), gt)
            plt.imsave(os.path.join(args.output_dir, fname[:-4] + '_output.png'), G_pred)

        print('id: %d, running psnr: %.4f, running ssim: %.4f'
              % (idx, np.mean(running_psnr), np.mean(running_ssim)))

    print('Dataset: %s, average psnr: %.4f, average ssim: %.4f'
          % (args.dataset, np.mean(running_psnr), np.mean(running_ssim)))



if __name__ == '__main__':

    # args.dataset = 'xzhang'
    # args.net_G = 'unet_512'
    # args.in_size = 512
    # args.ckptdir = 'checkpoints'

    # args.dataset = 'syn3-all'
    # args.net_G = 'unet_512'
    # args.in_size = 512
    # args.ckptdir = 'checkpoints'

    # args.dataset = 'bdn'
    # args.net_G = 'unet_256'
    # args.in_size = 256
    # args.ckptdir = 'checkpoints'

    # args.save_output = True

    net_G = load_model(args)
    run_eval(args)





