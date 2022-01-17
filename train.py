import time
from options.train_options import TrainOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer


def print_loss(opt, model, visualizer, start_time, iter, epoch):
    losses = model.get_current_errors()
    t = (time.time() - start_time) / opt.batchSize
    visualizer.print_current_errors(epoch, iter, losses, t)
    if opt.display_id > 0:
        visualizer.plot_current_errors(iter, losses)


def train(opt):
    # create a model
    model = create_model(opt)

    # create a dataset
    dataset = Dataset.create_dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)
    # model = model.to()
    # create a visualizer
    visualizer = Visualizer(opt)
    # training flag
    keep_training = True

    opt.niter = opt.nepoch * len(dataset)
    opt.niter_decay = opt.nepoch_decay * len(dataset)

    max_iteration = opt.niter + opt.niter_decay
    epoch = 0
    total_iteration = opt.iter_count

    # training process
    while (keep_training):
        epoch_start_time = time.time()
        epoch += 1
        print('\n Training epoch: %d' % epoch)

        # model.net_G.use_gt_mask = epoch < opt.nepoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(data)
            model.optimize_parameters()

            # print training loss and save logging information to the disk
            if i % opt.print_iters_freq == 0:
                print_loss(opt, model, visualizer, iter_start_time, i, epoch)

            if total_iteration > max_iteration:
                keep_training = False
                break

        print_loss(opt, model, visualizer, epoch_start_time, total_iteration, epoch)

        # display images on visdom and save images
        if epoch % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)
            if hasattr(model, 'distribution'):
                visualizer.plot_current_distribution(model.get_current_dis())

        # evaluate
        if epoch % opt.eval_freq == 0:
            model.eval()
            if hasattr(model, 'eval_metric_name'):
                eval_results = model.get_current_eval_results()
                visualizer.print_current_eval(epoch, total_iteration, eval_results)
                if opt.display_id > 0:
                    visualizer.plot_current_score(total_iteration, eval_results)

        # save the latest model every <save_latest_freq> iterations to the disk
        if epoch % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
            model.save_networks('latest')

        # save the model every <save_iter_freq> iterations to the disk
        if epoch % opt.save_freq == 0:
            print('saving the model of epoch %d' % epoch)
            model.save_networks(epoch)

        model.update_learning_rate()

        print('\nEnd training')


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    train(opt)
