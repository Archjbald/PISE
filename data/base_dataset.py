import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.transforms.functional as F
import torchvision.transforms as T
from util import pose_utils
from PIL import Image
import pandas as pd
import torch
import math
import numbers


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--angle', type=float, default=False, help="Max rotate angle in degrees")
        parser.add_argument('--shift', type=float, default=False, help="Max shift in % of full size")
        parser.add_argument('--scale', type=float, default=False, help="Min scale in %, max is 1")
        parser.add_argument('--color', action='store_true', help='Color augment')

        parser.add_argument('--augment_proba', type=float, default=0.3)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.name_pairs, self.par_dir = self.get_paths(opt)
        self.size = len(self.name_pairs)
        self.dataset_size = min(opt.max_dataset_size, self.size)
        self.class_num = 8

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        transform_list = []
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')
        self.actors_list = self.init_actors_list()

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        par_paths = []
        assert False, "A subclass of MarkovAttnDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths, par_paths

    def __getitem__(self, index):
        if self.opt.phase == 'train' and self.opt.random:
            actor = random.choice(list(self.actors_list.keys()))
            for _ in range(10):
                if len(self.actors_list[actor]) < 2:
                    actor = random.choice(list(self.actors_list.keys()))
                else:
                    break
            P1_name, P2_name = random.sample(self.actors_list[actor], 2)
        else:
            if self.opt.phase == 'train' or self.opt.random:
                index = random.randint(0, self.size - 1)

            P1_name, P2_name = self.name_pairs[index]

        P1_path = os.path.join(self.image_dir, P1_name)  # person 1
        P2_path = os.path.join(self.image_dir, P2_name)  # person 2

        SPL1_path = os.path.join(self.par_dir, P1_name[:-4] + '.png')
        SPL2_path = os.path.join(self.par_dir, P2_name[:-4] + '.png')

        P1_img = Image.open(P1_path).convert('RGB')  # .crop(regions)
        P2_img = Image.open(P2_path).convert('RGB')  # .crop(regions)
        SPL1_img = Image.open(SPL1_path)  # .crop(regions)
        SPL2_img = Image.open(SPL2_path)  # .crop(regions)

        P1_img = self.pad_old(P1_img)
        P2_img = self.pad_old(P2_img)

        s1np = np.expand_dims(np.array(SPL1_img), -1)
        s2np = np.expand_dims(np.array(SPL2_img), -1)
        s1np = np.concatenate([s1np, s1np, s1np], -1)
        s2np = np.concatenate([s2np, s2np, s2np], -1)
        SPL1_img = Image.fromarray(np.uint8(s1np))
        SPL2_img = Image.fromarray(np.uint8(s2np))

        angle, shift, scale = self.getRandomAffineParam()
        P1_img = F.affine(P1_img, angle=angle, translate=shift, scale=scale, shear=0)  # , fillcolor=(128, 128, 128)
        SPL1_img = F.affine(SPL1_img, angle=angle, translate=shift, scale=scale, shear=0)  # , fillcolor=(128, 128, 128)
        center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        BP1 = self.obtain_bone(P1_name, affine_matrix)

        angle, shift, scale = self.getRandomAffineParam()
        # angle, shift, scale = angle * 0.2, (
        #     shift[0] * 0.5, shift[1] * 0.5), 1  # Reduce the deform parameters of the generated image

        P2_img = F.affine(P2_img, angle=angle, translate=shift, scale=scale, shear=0)  # , fillcolor=(128, 128, 128)
        SPL2_img = F.affine(SPL2_img, angle=angle, translate=shift, scale=scale, shear=0)  # , fillcolor=(128, 128, 128)
        center = (P2_img.size[0] * 0.5 + 0.5, P2_img.size[1] * 0.5 + 0.5)
        affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        BP2 = self.obtain_bone(P2_name, affine_matrix)

        # P1_img, P2_img = self.color_augment(P1_img, P2_img)
        P1 = self.trans(P1_img)
        P2 = self.trans(P2_img)

        SPL1_img = np.expand_dims(np.array(SPL1_img)[:, :, 0], 0)  # [:,:,40:-40] # 1*256*176
        SPL2_img = np.expand_dims(np.array(SPL2_img)[:, :, 0], 0)  # [:,:,40:-40]

        # print(SPL1_img.shape)
        # SPL1_img = SPL1_img.transpose(2,0)
        # SPL2_img = SPL2_img.transpose(2,0)
        _, h, w = SPL2_img.shape
        # print(SPL2_img.shape,SPL1_img.shape)
        num_class = self.class_num
        tmp = torch.from_numpy(SPL2_img).view(-1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL2_onehot = ones.view([h, w, num_class])
        # print(SPL2_onehot.shape)
        SPL2_onehot = SPL2_onehot.permute(2, 0, 1)

        tmp = torch.from_numpy(SPL1_img).view(-1).long()
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL1_onehot = ones.view([h, w, num_class])
        # print(SPL2_onehot.shape)
        SPL1_onehot = SPL1_onehot.permute(2, 0, 1)

        # print(SPL1.shape)
        SPL2 = torch.from_numpy(SPL2_img).long()

        return {'P1': P1, 'BP1': BP1, 'P2': P2, 'BP2': BP2, 'SPL1': SPL1_onehot, 'SPL2': SPL2_onehot, 'label_P2': SPL2,
                'P1_path': P1_name, 'P2_path': P2_name}

    def obtain_bone(self, name, affine_matrix):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose = pose_utils.cords_to_map(array, self.load_size, self.opt.old_size, affine_matrix)
        pose = np.transpose(pose, (2, 0, 1))
        pose = torch.Tensor(pose)
        return pose

    def __len__(self):
        return self.dataset_size

    def name(self):
        assert False, "A subclass of BaseDataset must override self.name"

    def getRandomAffineParam(self):
        angle = 0
        shift_x = 0
        shift_y = 0
        scale = 1

        is_train = self.opt.phase == 'train'
        if is_train:
            if self.opt.angle is not False and random.random() < self.opt.augment_proba:
                max_angle = abs(self.opt.angle)
                angle = int(np.random.uniform(low=-max_angle, high=max_angle))
            if self.opt.scale is not False and random.random() < self.opt.augment_proba:
                min_scale = min(self.opt.scale, 1.)
                scale = np.random.uniform(low=min_scale, high=1.)
            if self.opt.shift is not False and random.random() < self.opt.augment_proba:
                max_shift = int(self.opt.shift * self.opt.load_size)
                shift_x = int(np.random.uniform(low=-max_shift, high=max_shift))
                shift_y = int(np.random.uniform(low=-max_shift, high=max_shift))

        return angle, (shift_x, shift_y), scale

    def color_augment(self, *imgs):
        if not self.opt.color or not self.opt.phase == 'train':
            return imgs
        img_stack = Image.fromarray(np.vstack([np.array(img) for img in imgs]))

        if random.random() < self.opt.augment_proba:
            img_stack = F.to_grayscale(img_stack, num_output_channels=3)
        elif random.random() < self.opt.augment_proba:
            jitter = T.ColorJitter(brightness=.5, hue=.3)
            img_stack = jitter(img_stack)

        if random.random() < self.opt.augment_proba:
            img_stack = F.gaussian_blur(img_stack, kernel_size=(3 + 2 * random.randint(0, 3)))

        img_stack = np.array(img_stack)
        height = img_stack.shape[0] // len(imgs)
        return [Image.fromarray(img_stack[height * i:height * (i + 1)]) for i in range(len(imgs))]

    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
        #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
        #                              [     0                  0          1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = [math.radians(s) for s in shear]
        elif isinstance(shear, numbers.Number):
            shear = math.radians(shear)
            shear = [shear, 0]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix

    def get_affine_matrix(self, center, angle, translate, scale, shear):
        matrix_inv = self.get_inverse_affine_matrix(center, angle, translate, scale, shear)

        matrix_inv = np.matrix(matrix_inv).reshape(2, 3)
        pad = np.matrix([0, 0, 1])
        matrix_inv = np.concatenate((matrix_inv, pad), 0)
        matrix = np.linalg.inv(matrix_inv)
        return matrix

    def pad_old(self, img):
        height, width = self.opt.old_size
        array = np.array(img)

        pad_h = 0
        pad_w = 0
        array_pad = np.zeros([256, 256] + list(array.shape[2:]), dtype=array.dtype)
        if height < 256:
            pad_h = (256 - height) // 2
        if width < 256:
            pad_w = (256 - width) // 2

        array_pad[pad_h:256 - pad_h, pad_w:256 - pad_w] = array

        return Image.fromarray(array_pad)

    def init_actors_list(self):
        actors_list = {}
        for img in self.annotation_file.index:
            actor = img.split('_')[0]
            actors_list.setdefault(actor, []).append(img)

        return actors_list

