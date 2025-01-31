from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # training epoch
        parser.add_argument('--iter_count', type=int, default=1, help='the starting epoch count')
        parser.add_argument('--niter', type=int, default=5000000, help='# of iter with initial learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to decay learning rate to zero')
        parser.add_argument('--nepoch', type=int, default=400, help='# of epoch with initial learning rate')
        parser.add_argument('--nepoch_decay', type=int, default=400, help='# of epoch to decay learning rate to zero')

        # learning rate and loss weight
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy[lambda|step|plateau]')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['wgan-gp', 'hinge', 'lsgan'])

        # display the results
        parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
        parser.add_argument('--eval_freq', type=int, default=20, help='frequency of showing training results on screen')
        parser.add_argument('--print_iters_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1, help='frequency of saving the latest results')
        parser.add_argument('--save_freq', type=int, default=200, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results')

        parser.add_argument('--augment_D', action='store_true', help='use data_augment before discriminator')

        self.isTrain = True

        return parser
