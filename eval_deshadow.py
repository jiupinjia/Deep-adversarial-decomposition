import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse
import skimage.color as skcolor

import utils

import torch
import torchvision.transforms.functional as TF
import cyclegan_networks as cycnet

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='EVAL_DEREFLECTION')
parser.add_argument('--dataset', type=str, default='istd', metavar='str',
                    help='dataset name from [istd, srd], (default: istd)')
parser.add_argument('--in_size', type=int, default=256, metavar='N',
                    help='size of input image during eval')
parser.add_argument('--ckptdir', type=str, default='./checkpoints',
                    help='checkpoints dir (default: ./checkpoints)')
parser.add_argument('--net_G', type=str, default='unet_256', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_256)')
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

    rmse_all = 0
    m_pxl_all = 0
    rmse_shadow = 0
    m_pxl_shadow = 0
    rmse_bg = 0
    m_pxl_bg = 0

    if args.dataset == 'istd':
        datadir = r'./datasets/ISTD/test/test_A'
        val_dirs = glob.glob(os.path.join(datadir, '*.png'))
    elif args.dataset == 'srd':
        datadir = r'./datasets/SRD/Test/shadow'
        val_dirs = glob.glob(os.path.join(datadir, '*.jpg'))

    for idx in range(len(val_dirs)):

        this_dir = val_dirs[idx]

        if args.dataset == 'istd':
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(val_dirs[idx].replace('test_A', 'test_C'), cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(val_dirs[idx].replace('test_A', 'test_B'), cv2.IMREAD_COLOR)
            mask = cv2.resize(mask, (256, 256), cv2.INTER_NEAREST)
        elif args.dataset == 'srd':
            img_mix = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(val_dirs[idx].replace('shadow', 'shadow_free')[:-4] + '_free.jpg', cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            mask = np.ones([512, 512, 3])

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

        error_map = np.abs(skcolor.rgb2lab(G_pred) - skcolor.rgb2lab(gt))
        # error_map = np.abs(skcolor.rgb2lab(img_mix) - skcolor.rgb2lab(gt))
        rmse_all += np.sum(error_map)
        m_pxl_all += error_map.size / 3.0
        rmse_shadow += np.sum(error_map[mask>0])
        m_pxl_shadow += error_map[mask>0].size / 3.0
        rmse_bg += np.sum(error_map[mask==0])
        m_pxl_bg += error_map[mask==0].size / 3.0

        if args.save_output:
            fname = this_dir.split('\\')[-1]
            plt.imsave(os.path.join(args.output_dir, fname[:-4]+'_input.png'), img_mix)
            plt.imsave(os.path.join(args.output_dir, fname[:-4]+'_gt.png'), gt)
            plt.imsave(os.path.join(args.output_dir, fname[:-4]+'_output.png'), G_pred)

        print('id: %d, running rmse-shadow: %.4f, rmse-non-shadow: %.4f, rmse-all: %.4f'
              % (idx, rmse_shadow/m_pxl_shadow, rmse_bg/m_pxl_bg, rmse_all/m_pxl_all))

    print('Dataset: %s, average rmse-shadow: %.4f, rmse-non-shadow: %.4f, rmse-all: %.4f'
          % (args.dataset, rmse_shadow/m_pxl_shadow, rmse_bg/m_pxl_bg, rmse_all/m_pxl_all))



if __name__ == '__main__':

    # args.dataset = 'srd'
    # args.net_G = 'unet_512'
    # args.in_size = 512
    # args.ckptdir = 'checkpoints'

    # args.dataset = 'istd'
    # args.net_G = 'unet_256'
    # args.in_size = 256
    # args.ckptdir = 'checkpoints'

    # args.save_output = True

    net_G = load_model(args)
    run_eval(args)





