from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer
from itertools import islice
import numpy as np
import torch

if __name__ == '__main__':
    # get testing options
    opt = TestOptions().parse()
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    visualizer = Visualizer(opt)

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            visualizer.display_current_results(model.get_current_visuals(), i)
