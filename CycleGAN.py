import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

class CycleGAN():
    def __init__(self):
        self.opt = TestOptions().parse()  # get test options

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        if opt.eval:
            model.eval()            # regular setup: load and print networks; create schedulers

    def _configure_test_options(self):
        """
        This function sets pre-configured options for model testing
        """
        self.opt.num_threads = 0   # test code only supports num_threads = 0
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        self.opt.name = model