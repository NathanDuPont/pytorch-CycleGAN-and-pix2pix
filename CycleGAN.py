import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

class CycleGAN():
    def __init__(self, model_name: str):
        """
        This function initializes a new CycleGAN model

        @param model_name Style transfer model to use. Examples include:
            style_monet_pretrained

        """
        self.opt = TestOptions().parse()  # get test options

        self._configure_test_options()

        self.opt.name = model_name

        self.model = create_model(self.opt)      # create a model given opt.model and other options
        self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers
        if self.opt.eval:
            self.model.eval()            # regular setup: load and print networks; create schedulers

    def _configure_test_options(self):
        """
        This function sets pre-configured options for model testing
        """
        self.opt.num_threads = 0   # test code only supports num_threads = 0
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    def set_input(self, input_data):
        """
        This function is used to provide custom input to the model. Data 
        should be in the form of a dense array, or a OpenCV Mat object
        """
        self.model.set_input(input_data)

    def run_inference(self):
        """
        This function runs a single pass of inference on the model

        @returns Dictionary containing image data from the "real" and
            "fake" (post style transfer) image
        """
        self.model.test()
        return self.model.get_current_visuals()