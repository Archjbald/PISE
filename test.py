from itertools import islice
import os
from util import html

import numpy as np
import torch

from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer
from util.util import get_kps


class HPEAnnots:
    def __init__(self):
        self.annots = {'x': [], 'y': []}
        # Picture format coordinates
        self.img_paths = []

    def add_annots(self, bps, img_paths=None):
        kps, v = get_kps(bps)

        vis = v[:, :, None]
        kps = kps * vis + (~vis) * -1
        self.annots['x'] += kps[:, :, 1].tolist()
        self.annots['y'] += kps[:, :, 0].tolist()

        if img_paths:
            self.img_paths += img_paths

    def save(self, path, name=None):
        annots_csv = "name:keypoints_y:keypoints_x\n"
        annots_csv += '\n'.join(
            [f'{img}:{self.annots["y"][i]}:{self.annots["x"][i]}' for i, img in enumerate(self.img_paths)])

        with open(os.path.join(path, f"annots_{name}.csv"), 'w') as f:
            f.write(annots_csv)


if __name__ == '__main__':
    # get testing options
    opt = TestOptions().parse()
    opt.no_html = True
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.which_iter}")

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_iter))
    hpe_annots = HPEAnnots()

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            # visualizer.display_current_results(model.get_current_visuals(), i, save_result=True)
            img_paths = model.get_image_paths()
            hpe_annots.add_annots(model.input_BP1, img_paths=[ip.split('___')[1] for ip in img_paths])

            visuals = model.get_current_visuals_test()
            visualizer.save_images(webpage, visuals, img_paths)
            # visualizer.save_images(model.get_current_visuals(), i)

    webpage.save()
    hpe_annots.save(webpage.web_dir, name="real")
