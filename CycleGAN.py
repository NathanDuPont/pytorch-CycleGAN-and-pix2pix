import torch
import torchvision
import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

class CycleGAN():
    def __init__(self, model_name: str):
        """
        This function initializes a new CycleGAN model

        @param model_name Style transfer model to use, as a string. Examples include:
            style_monet_pretrained
            style_vangogh_pretrained
            style_ukiyoe_pretrained
            style_cezanne_pretrained
        """
        self.opt = TestOptions().parse()  # get test options

        self._configure_test_options()

        self.opt.name = model_name

        self.input_dir = None
        self.img_extensions = ['.jpg', '.jpeg', '.png']

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

    def _format_image(self, image_data):
        """
        This function formats image data to match what the CycleGAN model
        is expecting. Image data will be returned in a dictionary of the
        following format:

        dict = {
            "A": [image data]
            "A_paths": None
        }

        @param image_data: Accepts image data in a numpy array
        @returns: Dictionary in the CycleGAN image format
        """
        # Add an extra dimension to the array to account for batch size
        image_data = np.array([image_data])

        # Format the axes correctly to match desired order:
        # Batch size x Channels x Height x Width
        image_data = image_data.transpose([0, 3, 1, 2])

        # Format the data as a tensor
        image_tensor = torch.FloatTensor(image_data)

        return { "A": image_tensor, "A_paths": None }

    def set_model_input(self, input_data):
        """
        This function is used to provide custom input directory to the model. Data 
        should be in the form of a dense array, or a OpenCV Mat object
        
        @param input_data: Image input data in the form of a NumPy array
        """
        # Set the input to the model
        self.model.set_input(self._format_image(input_data))

    def run_inference(self):
        """
        This function runs a single pass of inference on the model

        @returns: Dictionary containing image data from the "real" and
            "fake" (post style transfer) image
        """
        # Run a single pass of inference
        self.model.test()

        # Return the current visuals
        return self.model.get_current_visuals()