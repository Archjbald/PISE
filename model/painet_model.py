import os
from collections import OrderedDict

import torch
import torch.nn as nn
import itertools
import numpy as np

import torch.nn.functional as F

from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import util
from data.augment_pipe import AugmentPipe


class Painet(BaseModel):
    def name(self):
        return "parsing and inpaint network"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        # parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for the Sampling Correctness loss')
        parser.add_argument('--lambda_style', type=float, default=200.0, help='weight for the VGG19 style loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for the VGG19 content loss')
        parser.add_argument('--lambda_regularization', type=float, default=30.0,
                            help='weight for the affine regularization loss')

        parser.add_argument('--use_spect_g', action='store_false',
                            help="whether use spectral normalization in generator")
        parser.add_argument('--use_spect_d', action='store_false',
                            help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")

        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['par', 'par1']
        self.only_mask = opt.mask
        if not self.only_mask:
            self.loss_names = ['app_gen', 'content_gen', 'style_gen',  # 'reg_gen',
                               'ad_gen', 'dis_img_gen', ] + self.loss_names

        self.visual_names = ['input_P1', 'input_P2', 'img_gen']
        self.model_names = ['G']
        if not self.only_mask:
            self.model_names += ['D']

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids) > 0 \
            else torch.FloatTensor

        # define the generator
        self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64,
                                      use_spect=opt.use_spect_g, norm='instance', activation='LeakyReLU',
                                      use_gt=opt.use_gt_mask, only_mask=opt.mask)

        # define the discriminator 
        if not self.only_mask:
            self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)

        trained_list = ['parnet']
        for k, v in self.net_G.named_parameters():
            flag = False
            for i in trained_list:
                if i in k:
                    flag = True
                    break
            if flag:
                # v.requires_grad = False
                # print(k)
                pass

        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.Vggloss = external_function.VGGLoss().to(opt.device)
            self.parLoss = CrossEntropyLoss2d()  # torch.nn.BCELoss()

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                filter(lambda p: p.requires_grad, self.net_G.parameters())),
                lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G)

            if not self.only_mask:
                self.optimizer_D = torch.optim.Adam(itertools.chain(
                    filter(lambda p: p.requires_grad, self.net_D.parameters())),
                    lr=opt.lr * opt.ratio_g2d, betas=(0.9, 0.999))
                self.optimizers.append(self.optimizer_D)

                if opt.augment_D:
                    self.augment_D = AugmentPipe(blit=1., geom=1., color=1., deform=True)
                else:
                    self.augment_D = None

        # load the pre-trained model and schedulers
        self.setup(opt)

    def set_input(self, input):
        # move to GPU and change data types
        self.input = input
        input_P1, input_BP1, input_SPL1 = input['P1'], input['BP1'], input['SPL1']
        input_P2, input_BP2, input_SPL2, label_P2 = input['P2'], input['BP2'], input['SPL2'], input['label_P2']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_SPL1 = input_SPL1.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_P2 = input_P2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0], non_blocking=True)
            self.input_SPL2 = input_SPL2.cuda(self.gpu_ids[0], non_blocking=True)
            self.label_P2 = label_P2.cuda(self.gpu_ids[0], non_blocking=True)

            # np.save(f'{time.time()}.npy', self.label_P2.clone().cpu().numpy())

        self.image_paths = []
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '___' + input['P2_path'][i])

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w - 1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h - 1)
        grid = torch.stack([x, y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2 * grid - 1
        flow = 2 * flow / torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid + flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample

    def test(self, save=False):
        """Forward function used in test time"""
        self.img_gen, self.loss_reg, self.parsav = self.net_G(self.input_P1, self.input_P2, self.input_BP1,
                                                              self.input_BP2,
                                                              self.input_SPL1, self.input_SPL2)
        ## test flow ##
        if save:
            self.save_results(self.img_gen, data_name='vis')
            if (self.opt.save_input or self.opt.phase == 'val') and not self.opt.mask:
                self.save_results(self.input_P1, data_name='ref')
                self.save_results(self.input_P2, data_name='gt')
                result = torch.cat([self.input_P1, self.img_gen, self.input_P2], 3)
                self.save_results(result, data_name='all')

    def forward(self):
        """Run forward processing to get the inputs"""
        self.img_gen, self.loss_reg, self.parsav = self.net_G(self.input_P1, self.input_P2, self.input_BP1,
                                                              self.input_BP2, self.input_SPL1, self.input_SPL2)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        if self.augment_D is not None:
            real = self.augment_D(real)
            fake = self.augment_D(fake)
        D_real = netD(real)
        if self.augment_D is not None:
            self.augment_D.update_p(D_real.sign())
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        # print(self.input_P2.shape, self.img_gen.shape)
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, self.input_P2, self.img_gen)

    def backward_G(self):
        """Calculate training loss for the generator"""
        # parsing loss
        label_P2 = self.label_P2.squeeze(1).long()
        # print(self.input_SPL2.min(), self.input_SPL2.max(), self.parsav.min(), self.parsav.max())
        self.loss_par = self.parLoss(self.parsav, label_P2)  # * 20.
        self.loss_par1 = self.L1loss(self.parsav, self.input_SPL2) * 100

        if not self.only_mask:
            # Calculate regularzation loss to make transformed feature and target image feature in the same latent space
            self.loss_reg_gen = self.loss_reg * self.opt.lambda_regularization

            # Calculate l1 loss
            loss_app_gen = self.L1loss(self.img_gen, self.input_P2)
            self.loss_app_gen = loss_app_gen * self.opt.lambda_rec

            # Calculate GAN loss
            base_function._freeze(self.net_D)
            D_fake = self.net_D(self.img_gen)
            self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

            # Calculate perceptual loss
            loss_content_gen, loss_style_gen = self.Vggloss(self.img_gen, self.input_P2)
            self.loss_style_gen = loss_style_gen * self.opt.lambda_style
            self.loss_content_gen = loss_content_gen * self.opt.lambda_content

        total_loss = 0

        for name in self.loss_names:
            if name != 'dis_img_gen':
                # print(getattr(self, "loss_" + name))
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        self.forward()
        if not self.only_mask:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        errors = BaseModel.get_current_errors(self)
        if self.augment_D is not None:
            errors['p'] = self.augment_D.p
            errors['p_adjust'] = self.augment_D.adjust
        return errors

    def get_current_visuals_test(self):
        nbj = self.opt.structure_nc
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        inputs_P1 = util.tensors2ims(self.input_P1.data)
        inputs_P2 = util.tensors2ims(self.input_P2.data)
        fakes_P2 = util.tensors2ims(self.img_gen.data)

        inputs_BP1 = util.draw_poses_from_maps(self.input_BP1[:, :nbj].data)
        inputs_BP2 = util.draw_poses_from_maps(self.input_BP2[:, :nbj].data)
        imgs = [[inputs_P1[i], inputs_BP1[i][0], inputs_P2[i], inputs_BP2[i][0], fakes_P2[i]] for i in
                range(self.opt.batchSize)]
        viss = [np.zeros((height, width * len(imgs[0]), 3)).astype(np.uint8) for _ in inputs_P1]  # h, w, c
        for j, vis in enumerate(viss):
            for i, img in enumerate(imgs[j]):
                vis[:, width * i:width * (i + 1), :] = img

        ret_visuals = [OrderedDict([('vis', vis)]) for vis in viss]

        return ret_visuals


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        return self.nll_loss(self.softmax(inputs), targets)
