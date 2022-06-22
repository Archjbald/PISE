# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import random

import torch
import numpy as np
import torchvision.transforms
from torchvision.transforms import functional as f
import torch.nn.functional as F

# from torch_utils.ops import grid_sample_gradfix


_constant_cache = dict()

from PIL import Image
import numpy as np


def disp_tensor(tens):
    array = tens.clone().cpu()
    array = array.detach().numpy()
    if array.ndim > 3:
        array = array[0]
    if array.ndim > 2:
        if array.shape[0] > 3:
            array = array.max(axis=0, keepdims=True)
        elif array.shape[0] == 3:
            array = (array + 1) / 2
    else:
        array = array[None]
    if array.shape[0] == 1:
        array = np.tile(array, (3, 1, 1))
    array = array.transpose((1, 2, 0))
    Image.fromarray((array * 255).astype(np.uint8)).show()


def get_rot_mat(theta):
    theta = np.radians(theta)
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, angle, align_corners=False):
    dtype = x.dtype
    device = x.device
    x = x[None, ...] + 1
    rot_mat = get_rot_mat(angle)[None, ...].type(dtype).to(device).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=align_corners).type(dtype)
    x_rot = F.grid_sample(x, grid, align_corners=align_corners)
    return x_rot[0] - 1


def apply_transfo_random(images, transfo, p, classes=(None,)):
    # Apply transformation to mini batch, individually
    trans = (torch.rand([len(images)], device=images.device))[:, None, None, None]
    p_slice = p / len(classes)
    trans_images = images
    for i, param in enumerate(classes):
        try:
            if param is not None:
                param = list(param)
        except TypeError:
            param = [param, ]
        class_trans = (trans >= (1 - p) + i * p_slice) * (trans < (1 - p) + (i + 1) * p_slice)
        trans_images = (transfo(images) if param is None else transfo(images,
                                                                      *param)) * class_trans + trans_images * ~class_trans
    return trans_images


def translate(x, params):
    h, w = x.shape[-2:]
    params = [int(p) for p in params]
    x_trans = x[:, max(0, -params[0]):min(h, h - params[0]), max(0, -params[1]):min(w, w - params[1])]

    new_x = torch.ones_like(x) * -1
    h = max(0, params[0])
    w = max(0, params[1])
    new_x[:, h:h + x_trans.shape[-2],
          w: w + x_trans.shape[-1]] = x_trans
    return new_x


class AugmentPipe(torch.nn.Module):
    def __init__(self, blit=0, geom=0, color=0, deform=True):
        super(AugmentPipe, self).__init__()
        self.p = 0.
        self.adjust = 0.

        self.blit = blit
        self.geom = geom
        self.color = color

        self.deform = deform

        self.buff = []

    def update_p(self, val):
        self.buff.append(val.detach().clone().cpu())
        if len(self.buff) >= 4:
            self.adjust = np.sign(torch.cat(self.buff).mean() - 0.6) * len(self.buff) * len(self.buff[0]) / (100 * 1000)
            self.p = min(1., max(0., self.p + self.adjust))
            self.buff = []

    def forward(self, images):
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device

        fill = -1

        if self.blit > 0:
            # Apply x-flip with probability (xflip * strength).
            images = apply_transfo_random(images, f.hflip, self.blit * self.p)

            # Apply 90 degree rotations with probability (rotate90 * strength).
            images = apply_transfo_random(images, f.rotate, self.blit * self.p, (90, 180, 270))

            # Apply integer translation with probability (xint * strength).
            for i in range(len(images)):
                if random.random() < self.blit * self.p * self.deform:
                    # images[i] = f.affine(img,
                    #                      translate=[0.125 * d * (random.random() * 2 - 1) for d in images.shape[-2:]],
                    #                      angle=0, scale=1., shear=0., fill=fill)
                    images[i] = translate(images[i], [0.125 * d * (random.random() * 2 - 1) for d in images.shape[-2:]])

        if self.geom > 0:
            # Apply isotropic scaling with probability (scale * strength).
            for i in range(len(images)):
                img = images[i]
                if random.random() < self.geom * self.p * self.deform:
                    scaled_img = f.resize(img, int(min(height, width) * (1 - 0.2 * random.random())))
                    h = (height - scaled_img.shape[-2]) // 2
                    w = (width - scaled_img.shape[-1]) // 2
                    new_img = torch.ones_like(img) * -1
                    new_img[:, h:h + scaled_img.shape[-2],
                    w: w + scaled_img.shape[-1]] = scaled_img
                    images[i] = new_img

                # Apply pre-rotation with probability p_rot.
                if random.random() < self.geom * self.p * self.deform:
                    images[i] = rot_img(img, angle=int(45 * random.random()))
                    # images[i] = f.rotate(img, angle=int(45 * random.random()), fill=fill)

                # Apply anisotropic scaling with probability (aniso * strength).
                if random.random() < self.geom * self.p * self.deform:
                    scaled_img = f.resize(img, [int((1 - 0.2 * random.random()) * d) for d in img.shape[-2:]])
                    h = (height - scaled_img.shape[-2]) // 2
                    w = (width - scaled_img.shape[-1]) // 2
                    new_img = torch.ones_like(img) * -1
                    new_img[:, h:h + scaled_img.shape[-2],
                    w: w + scaled_img.shape[-1]] = scaled_img
                    images[i] = new_img

                # Apply post-rotation with probability p_rot.
                if random.random() < self.geom * self.p * self.deform:
                    images[i] = rot_img(img, angle=int(45 * random.random()))
                    # images[i] = f.rotate(img, angle=int(45 * random.random()), fill=fill)

                # Apply fractional translation with probability (xfrac * strength).
                if random.random() < self.geom * self.p * self.deform:
                    # images[i] = f.affine(img,
                    #                      translate=[0.125 * d * random.random() for d in images.shape[-2:]],
                    #                      angle=0, scale=1., shear=0., fill=fill)
                    images[i] = translate(img, [0.125 * d * random.random() for d in images.shape[-2:]])

                # Color transformation
                img = images[i] + 1 / 2
                if random.random() < self.color * self.p:
                    for p in range(4):
                        color_jitter = torchvision.transforms.ColorJitter(
                            brightness=0.2 * (p == 0), contrast=0.5 * (p == 1), saturation=0.5 * (p == 3),
                            hue=0.2 * (p == 2))
                        img = color_jitter(img)

                images[i] = img * 2 - 1

        return images
