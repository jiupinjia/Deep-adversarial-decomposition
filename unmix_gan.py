import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import copy
import os

import utils
import loss
import cyclegan_networks as cycnet

import torch
torch.cuda.current_device()
import torchvision.models as torchmodels
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn


# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class UnmixGAN():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders
        self.net_D1 = cycnet.define_D(input_nc=6, ndf=64, netD='n_layers', n_layers_D=2).to(device)
        self.net_D2 = cycnet.define_D(input_nc=6, ndf=64, netD='n_layers', n_layers_D=2).to(device)
        self.net_D3 = cycnet.define_D(input_nc=6, ndf=64, netD='n_layers', n_layers_D=3).to(device)
        self.net_G = cycnet.define_G(
            input_nc=3, output_nc=6, ngf=args.ngf, netG=args.net_G, use_dropout=False, norm='none').to(device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        self.optimizer_G = optim.Adam(
            self.net_G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D1 = optim.Adam(
            self.net_D1.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D2 = optim.Adam(
            self.net_D2.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D3 = optim.Adam(
            self.net_D3.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # define lr schedulers
        self.exp_lr_scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, step_size=args.exp_lr_scheduler_stepsize, gamma=0.1)
        self.exp_lr_scheduler_D1 = lr_scheduler.StepLR(
            self.optimizer_D1, step_size=args.exp_lr_scheduler_stepsize, gamma=0.1)
        self.exp_lr_scheduler_D2 = lr_scheduler.StepLR(
            self.optimizer_D2, step_size=args.exp_lr_scheduler_stepsize, gamma=0.1)
        self.exp_lr_scheduler_D3 = lr_scheduler.StepLR(
            self.optimizer_D3, step_size=args.exp_lr_scheduler_stepsize, gamma=0.1)

        # coefficient to balance loss functions
        self.lambda_L1 = args.lambda_L1
        self.lambda_adv = args.lambda_adv

        # based on which metric to update the "best" ckpt
        self.metric = args.metric

        # define some other vars to record the training states
        self.running_acc = []
        self.epoch_acc = 0
        if 'mse' in self.metric:
            self.best_val_acc = 1e9  # for mse, rmse, a lower score is better
        else:
            self.best_val_acc = 0.0  # for others (ssim, psnr), a higher score is better
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_num_epochs
        self.G_pred1 = None
        self.G_pred2 = None
        self.batch = None
        self.G_loss = None
        self.D_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.D1_fake_pool = utils.ImagePool(pool_size=50)
        self.D2_fake_pool = utils.ImagePool(pool_size=50)
        self.D3_fake_pool = utils.ImagePool(pool_size=50)

        # define the loss functions
        if args.pixel_loss == 'minimum_pixel_loss':
            self._pxl_loss = loss.MinimumPixelLoss(opt=1) # 1 for L1 and 2 for L2
        elif args.pixel_loss == 'pixel_loss':
            self._pxl_loss = loss.PixelLoss(opt=1)  # 1 for L1 and 2 for L2
        else:
            raise NotImplementedError('pixel loss function [%s] is not implemented', args.pixel_loss)
        self._gan_loss = loss.GANLoss(gan_mode='vanilla').to(device)
        self._exclusion_loss = loss.ExclusionLoss()
        self._kurtosis_loss = loss.KurtosisLoss()
        # enable some losses?
        self.with_d1d2 = args.enable_d1d2
        self.with_d3 = args.enable_d3
        self.with_exclusion_loss = args.enable_exclusion_loss
        self.with_kurtosis_loss = args.enable_kurtosis_loss

        # m-th epoch to activate adversarial training
        self.m_epoch_activate_adv = int(self.max_num_epochs / 20) + 1

        # output auto-enhancement?
        self.output_auto_enhance = args.output_auto_enhance

        # use synfake to train D?
        self.synfake = args.enable_synfake

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

        # visualize model
        if args.print_models:
            self._visualize_models()


    def _visualize_models(self):

        from torchviz import make_dot

        # visualize models with the package torchviz
        y = self.net_G(torch.rand(4, 3, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_G.named_parameters()))
        mygraph.render('G')
        y = self.net_D1(torch.rand(4, 6, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_D1.named_parameters()))
        mygraph.render('D1')
        y = self.net_D2(torch.rand(4, 6, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_D2.named_parameters()))
        mygraph.render('D2')
        y = self.net_D3(torch.rand(4, 6, 512, 512).to(device))
        mygraph = make_dot(y.mean(), params=dict(self.net_D3.named_parameters()))
        mygraph.render('D3')


    def _load_checkpoint(self):

        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('loading last checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(device)

            # update net_D1 states
            self.net_D1.load_state_dict(checkpoint['model_D1_state_dict'])
            self.optimizer_D1.load_state_dict(checkpoint['optimizer_D1_state_dict'])
            self.exp_lr_scheduler_D1.load_state_dict(
                checkpoint['exp_lr_scheduler_D1_state_dict'])
            self.net_D1.to(device)

            # update net_D2 states
            self.net_D2.load_state_dict(checkpoint['model_D2_state_dict'])
            self.optimizer_D2.load_state_dict(checkpoint['optimizer_D2_state_dict'])
            self.exp_lr_scheduler_D2.load_state_dict(
                checkpoint['exp_lr_scheduler_D2_state_dict'])
            self.net_D2.to(device)

            # update net_D3 states
            self.net_D3.load_state_dict(checkpoint['model_D3_state_dict'])
            self.optimizer_D3.load_state_dict(checkpoint['optimizer_D3_state_dict'])
            self.exp_lr_scheduler_D3.load_state_dict(
                checkpoint['exp_lr_scheduler_D3_state_dict'])
            self.net_D3.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d, %s)' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id, self.metric))
            print()

        else:
            print('training from scratch...')


    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'model_D1_state_dict': self.net_D1.state_dict(),
            'optimizer_D1_state_dict': self.optimizer_D1.state_dict(),
            'exp_lr_scheduler_D1_state_dict': self.exp_lr_scheduler_D1.state_dict(),
            'model_D2_state_dict': self.net_D2.state_dict(),
            'optimizer_D2_state_dict': self.optimizer_D2.state_dict(),
            'exp_lr_scheduler_D2_state_dict': self.exp_lr_scheduler_D2.state_dict(),
            'model_D3_state_dict': self.net_D3.state_dict(),
            'optimizer_D3_state_dict': self.optimizer_D3.state_dict(),
            'exp_lr_scheduler_D3_state_dict': self.exp_lr_scheduler_D3.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))


    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
        self.exp_lr_scheduler_D1.step()
        self.exp_lr_scheduler_D2.step()
        self.exp_lr_scheduler_D3.step()


    def _compute_acc(self):

        target1 = self.batch['gt1'].to(device).detach()
        target2 = self.batch['gt2'].to(device).detach()
        img1 = self.G_pred1.detach()
        img2 = self.G_pred2.detach()

        if self.metric == 'psnr':
            acc1 = 0.5*utils.cpt_psnr(img1, target1, PIXEL_MAX=1.0) + \
                   0.5*utils.cpt_psnr(img2, target2, PIXEL_MAX=1.0)
            acc2 = 0.5*utils.cpt_psnr(img1, target2, PIXEL_MAX=1.0) + \
                   0.5*utils.cpt_psnr(img2, target1, PIXEL_MAX=1.0)
            return max(acc1, acc2)
        elif self.metric == 'psnr_gt1':
            acc = utils.cpt_psnr(img1, target1, PIXEL_MAX=1.0)
            return acc
        elif self.metric == 'ssim':
            acc1 = 0.5*utils.cpt_ssim(img1, target1) + \
                   0.5*utils.cpt_ssim(img2, target2)
            acc2 = 0.5*utils.cpt_ssim(img1, target2) + \
                   0.5*utils.cpt_ssim(img2, target1)
            return max(acc1, acc2)
        elif self.metric == 'ssim_gt1':
            acc = utils.cpt_ssim(img1, target1)
            return acc
        elif self.metric == 'labrmse_gt1':
            acc = utils.cpt_labrmse(img1, target1)
            return acc
        else:
            raise NotImplementedError('metric method [%s] is not implemented' % self.metric)



    def _collect_running_batch_states(self):
        self.running_acc.append(self._compute_acc().item())

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        if np.mod(self.batch_id, 100) == 1 or self.batch_id == m-1:
            print('Is_training: %s. [%d,%d][%d,%d], G_loss: %.8f, D_loss: %.8f, running_acc: %.8f (%s),'
                  % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     self.G_loss.item(), self.D_loss.item(),
                     np.mean(self.running_acc), self.metric))

        if np.mod(self.batch_id, 1000) == 1 or self.batch_id == m-1:
            vis_input = utils.make_numpy_grid(self.batch['input'])
            vis_pred1 = utils.make_numpy_grid(self.G_pred1)
            vis_pred2 = utils.make_numpy_grid(self.G_pred2)
            if self.output_auto_enhance:
                vis_pred1 = vis_pred1*1.5
                vis_pred2 = vis_pred2*1.5
            vis = np.concatenate([vis_input, vis_pred1, vis_pred2], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)



    def _collect_epoch_states(self):

        self.epoch_acc = np.mean(self.running_acc)
        print('Is_training: %s. Epoch %d / %d, epoch_acc= %.8f (%s),' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc, self.metric))
        print()


    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        print('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        print()

        # update the best model (based on eval acc)
        if self.metric == 'labrmse_gt1':
            if self.epoch_acc < self.best_val_acc:
                # a lower score is better
                self.best_val_acc = self.epoch_acc
                self.best_epoch_id = self.epoch_id
                self._save_checkpoint(ckpt_name='best_ckpt.pt')
                print('*' * 10 + 'Best model updated!')
                print()
        else:
            if self.epoch_acc > self.best_val_acc:
                # a higher score is better
                self.best_val_acc = self.epoch_acc
                self.best_epoch_id = self.epoch_id
                self._save_checkpoint(ckpt_name='best_ckpt.pt')
                print('*' * 10 + 'Best model updated!')
                print()
        # update the best model (based on eval acc)



    def _clear_cache(self):
        self.running_acc = []


    def _forward_pass(self, batch):
        self.batch = batch
        img_in = batch['input'].to(device)
        y = self.net_G(img_in)
        self.G_pred1 = y[:, 0:3, :, :]
        self.G_pred2 = y[:, 3:, :, :]


    def _backward_D(self):

        self.D_loss = torch.tensor(0.0, requires_grad=True).to(device)
        img_in = self.batch['input'].to(device)
        gt1 = (self.batch['gt1']).to(device)
        gt2 = (self.batch['gt2']).to(device)

        if self.epoch_id >= self.m_epoch_activate_adv:

            if self.with_d1d2:
                # D1
                fake_cat = torch.cat((img_in, self.G_pred1), dim=1).detach()
                fake_cat = self.D1_fake_pool.query(fake_cat)
                D1_pred_fake = self.net_D1(fake_cat)
                real_cat = torch.cat((img_in, gt1), dim=1).detach()
                D1_pred_real = self.net_D1(real_cat)
                D1_adv_loss_fake = self._gan_loss(D1_pred_fake, False)
                D1_adv_loss_real = self._gan_loss(D1_pred_real, True)
                D1_adv_loss = 0.5*(D1_adv_loss_fake + D1_adv_loss_real)

                # D2
                fake_cat = torch.cat((img_in, self.G_pred2), dim=1).detach()
                fake_cat = self.D2_fake_pool.query(fake_cat)
                D2_pred_fake = self.net_D2(fake_cat)
                real_cat = torch.cat((img_in, gt2), dim=1).detach()
                D2_pred_real = self.net_D2(real_cat)
                D2_adv_loss_fake = self._gan_loss(D2_pred_fake, False)
                D2_adv_loss_real = self._gan_loss(D2_pred_real, True)
                D2_adv_loss = 0.5*(D2_adv_loss_fake + D2_adv_loss_real)

                self.D_loss += self.lambda_adv * (D1_adv_loss + D2_adv_loss)

            if self.with_d3:
                # D3
                fake_cat = torch.cat((self.G_pred1, self.G_pred2), dim=1).detach()
                fake_cat = self.D3_fake_pool.query(fake_cat)
                if self.synfake:
                    fake_cat = utils.insert_synfake(fake_cat, self.batch)

                fake_cat = nn.functional.interpolate(fake_cat, [64, 64])
                D3_pred_fake = self.net_D3(fake_cat)
                real_cat = torch.cat((gt1, gt2), dim=1).detach()
                real_cat = nn.functional.interpolate(real_cat, [64, 64])
                D3_pred_real = self.net_D3(real_cat)
                D3_adv_loss_fake = self._gan_loss(D3_pred_fake, False)
                D3_adv_loss_real = self._gan_loss(D3_pred_real, True)
                D3_adv_loss = 0.5 * (D3_adv_loss_fake + D3_adv_loss_real)

                self.D_loss += self.lambda_adv*D3_adv_loss

        self.D_loss.backward()


    def _backward_G(self):

        pixel_loss = self._pxl_loss.forward(
            batch=self.batch, G_pred1=self.G_pred1, G_pred2=self.G_pred2)
        G_adv_loss = torch.tensor(0.0, requires_grad=True).to(device)
        exclusion_loss = torch.tensor(0.0, requires_grad=True).to(device)
        kurtosis_loss = torch.tensor(0.0, requires_grad=True).to(device)

        if self.with_exclusion_loss:
            exclusion_loss = self._exclusion_loss.forward(
                G_pred1=self.G_pred1, G_pred2=self.G_pred2)

        if self.with_kurtosis_loss:
            kurtosis_loss = self._kurtosis_loss.forward(
                G_pred1=self.G_pred1, G_pred2=self.G_pred2)

        if self.epoch_id >= self.m_epoch_activate_adv:

            if self.with_d1d2:
                img_in = self.batch['input'].to(device)
                fake_cat = torch.cat((img_in, self.G_pred1), dim=1)
                D1_pred_fake = self.net_D1(fake_cat)
                fake_cat = torch.cat((img_in, self.G_pred2), dim=1)
                D2_pred_fake = self.net_D2(fake_cat)
                G_adv_loss += self._gan_loss(D1_pred_fake, True) + \
                             self._gan_loss(D2_pred_fake, True)

            if self.with_d3:
                fake_cat = torch.cat((self.G_pred1, self.G_pred2), dim=1)
                if self.synfake:
                    fake_cat = utils.insert_synfake(fake_cat, self.batch)
                fake_cat = nn.functional.interpolate(fake_cat, [64, 64])
                D3_pred_fake = self.net_D3(fake_cat)
                G_adv_loss += self._gan_loss(D3_pred_fake, True)

        self.G_loss = self.lambda_L1*pixel_loss + \
                      self.lambda_adv*G_adv_loss + \
                      2*exclusion_loss + \
                      kurtosis_loss
        self.G_loss.backward()



    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            self.net_D1.train()  # Set model to training mode
            self.net_D2.train()  # Set model to training mode
            self.net_D3.train()  # Set model to training mode
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update D1 and D2 and D3
                utils.set_requires_grad(self.net_D1, True)
                utils.set_requires_grad(self.net_D2, True)
                utils.set_requires_grad(self.net_D3, True)
                self.optimizer_D1.zero_grad()
                self.optimizer_D2.zero_grad()
                self.optimizer_D3.zero_grad()
                self._backward_D()
                self.optimizer_D1.step()
                self.optimizer_D2.step()
                self.optimizer_D3.step()
                # update G
                utils.set_requires_grad(self.net_D1, False)
                utils.set_requires_grad(self.net_D2, False)
                utils.set_requires_grad(self.net_D3, False)
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()
            self._update_lr_schedulers()

            ################## Eval ##################
            ##########################################
            print('Begin evaluation...')
            self._clear_cache()
            self.is_training = False
            # Set model to evaluate mode
            self.net_G.eval()
            self.net_D1.eval()
            self.net_D2.eval()
            self.net_D3.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_checkpoints()




