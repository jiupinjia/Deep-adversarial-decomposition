import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils
import unmix_gan as um

# settings
parser = argparse.ArgumentParser(description='ZZX UNMIXING')
parser.add_argument('--dataset', type=str, default='dogsflowers', metavar='str',
                    help='dataset name from [dogsflowers, cifar10, mnist, imagenet, '
                         'rain100h, rain100l, rain800, did-mdn, rain1400'
                         'bdn, syn3-defocused, syn3-focused, syn3-ghosting, syn3-all, xzhang,'
                         'istd, srd] (default: dogsflowers)')
parser.add_argument('--net_G', type=str, default='unet_128', metavar='str',
                    help='net_G: unet_512, unet_256 or unet_128 or unet_64 (default: unet_128)')
parser.add_argument('--ngf', type=int, default=64, metavar='N',
                    help='number of base filters in G')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--pixel_loss', type=str, default='minimum_pixel_loss', metavar='str',
                    help='type of pixel loss from [minimum_pixel_loss, pixel_loss] (default: minimum_pixel_loss)')
parser.add_argument('--print_models', action='store_true', default=False,
                    help='visualize and print networks')
parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoints', metavar='str',
                    help='dir to save checkpoints (default: ./checkpoints)')
parser.add_argument('--vis_dir', type=str, default=r'./val_out', metavar='str',
                    help='dir to save results during training (default: ./val_out)')
parser.add_argument('--enable_d1d2', action='store_true', default=False,
                    help='imporve the G by adversarial training')
parser.add_argument('--enable_d3', action='store_true', default=False,
                    help='imporve the G by adversarial training')
parser.add_argument('--enable_synfake', action='store_true', default=False,
                    help='to use synfake sample for training D')
parser.add_argument('--enable_exclusion_loss', action='store_true', default=False,
                    help='imporve the G by using exclusion loss')
parser.add_argument('--enable_kurtosis_loss', action='store_true', default=False,
                    help='imporve the G by using kurtosis loss')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--exp_lr_scheduler_stepsize', type=int, default=100,
                    help='every m steps to expositionally reduce lr to 0.1x')
parser.add_argument('--max_num_epochs', type=int, default=200, metavar='N',
                    help='max number of training epochs (default 200)')
parser.add_argument('--lambda_L1', type=float, default=1,
                    help='coefficient on L1 loss to balance with other losses')
parser.add_argument('--lambda_adv', type=float, default=1e-4,
                    help='coefficient on adv loss to balance with other losses')
parser.add_argument('--output_auto_enhance', action='store_true', default=False,
                    help='to enhance the output images')
parser.add_argument('--mix_mode', type=str, default='linear',
                    help='mode for generating mixed images, from [linear, nonlinear] (default: linear) ')
parser.add_argument('--metric', type=str, default='psnr',
                    help='metric to update ckpt, from [psnr, ssim, psnr_gt1, ssim_gt1, labrmse_gt1] (default: psnr)')
args = parser.parse_args()


if __name__ == '__main__':

    # For easy debug
    # args.dataset = 'mnist'
    # args.metric = 'psnr'
    # args.net_G = 'unet_64'
    # args.pixel_loss = 'minimum_pixel_loss'
    # args.max_num_epochs = 200
    # args.batch_size = 2
    # args.enable_d1d2 = True
    # args.enable_d3 = True
    # args.enable_synfake = True

    dataloaders = utils.get_loaders(args)

    # you may run the following if you want to check whether the data is loading correctly
    # for i in range(100):
    #     data = next(iter(dataloaders['train']))
    #     vis_input = utils.make_numpy_grid(data['input'])
    #     vis_pred1 = utils.make_numpy_grid(data['gt1'])
    #     vis_pred2 = utils.make_numpy_grid(data['gt2'])
    #     vis = np.concatenate([vis_input, vis_pred1, vis_pred2], axis=0)
    #     vis = np.clip(vis, a_min=0.0, a_max=1.0)
    #     plt.imshow(vis)
    #     plt.show()

    unmix = um.UnmixGAN(args=args, dataloaders=dataloaders)
    unmix.train_models()



