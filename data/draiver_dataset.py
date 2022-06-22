import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from util import pose_utils
import numpy as np
import torch


class DrAIverDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(dataroot="./dataset/draiver")
        parser.set_defaults(old_size=(192, 256))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        pairLst = os.path.join(root, 'draiver-pairs-%s.csv' % phase)  # 'draiver-pairs-%s.csv' % phase)
        #        pairLst = os.path.join(root, 'arbf_pres.csv')
        name_pairs = self.init_categories(pairLst)

        image_dir = os.path.join(root, '%s' % phase)
        bonesLst = os.path.join(root, 'draiver-annotation_%s.csv' % phase)  # 'draiver-annotation-%s.csv' % phase)
        par_dir = os.path.join(root, '%sSPL8' %phase)
        return image_dir, bonesLst, name_pairs, par_dir

    def init_categories(self, pairLst, take_dripe=True):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            if ('vp' in pair[0] and 'vp' in pair[1]) or take_dripe:
                pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs

    def init_actors_list(self):
        actors_list = {}
        for img in self.annotation_file.index:
            actor = '-'.join(img.split('_')[0].split('-')[:2 if img[:10] == "Transpolis" else 1])
            actors_list.setdefault(actor, []).append(img)

        return actors_list

    def name(self):
        return "DrAIverDataset"
