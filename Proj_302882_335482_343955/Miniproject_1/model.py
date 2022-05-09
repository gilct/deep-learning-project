import torch
import torch.nn as nn
from .others.models import *

# Since grad is disabled for minipart 2
# just enable it here to ensure that it is
# always enabled for part 1 (which it 
# should be). Otherwise it will cause 
# errors when tests load model.py from 
# part 1 after having previously loaded 
# model.py from part 2
torch.set_grad_enabled(True)

class Model():
    """
    Model class

    Attributes
    ----------
    batch_size : int
        the size of the batches to use during training
    model : ConvRes
        the network to use for training and predicting
    optimizer : torch.optim.Adam
        the optimizer to use during training
    criterion : nn.MSELoss
        the criterion to use during training

    Methods
    -------
    load_pretrained_model()
        Load a pretrained model
    train(train_input, train_target)
        Runs the training procedure
    predict(test_input)
        Generates a prediction on the input
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self) -> None:
        """Model constructor"""
        self.batch_size = 100
        self.model = ConvRes().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def load_pretrained_model(self) -> None:
        """Loads the parameters saved in bestmodel.pth into the model"""
        BESTMODEL_PATH = "Proj_302882_335482_343955/Miniproject_1/bestmodel.pth"
        # TODO: maybe just pull to cpu at the end of training
        state_dict = torch.load(BESTMODEL_PATH, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        

    def train(self, train_input, train_target, num_epochs) -> None:
        """Runs the training procedure

        Parameters
        ----------
        train_input: (N, C, H, W), torch.tensor
            Tensor containing a noisy version of the images.
        train_target: (N, C, H, W), torch.tensor
            Tensor containing another noisy version of the same 
            images, which only differs from the input by their noise.
        """

        noisy_imgs_1 = (train_input  / 255.0).float().to(self.device)
        noisy_imgs_2 = (train_target  / 255.0).float().to(self.device)

        # Not strictly necessary but this is what the state should be
        # anyways so just enforce it 
        self.model.train()
        with torch.enable_grad():
            for e in range(num_epochs):
                item = f'\r\nTraining epoch {e+1}/{num_epochs}...'
                print(item, sep=' ', end='', flush=True)
                for b in range(0, noisy_imgs_1.size(0), self.batch_size):
                    output = self.model(noisy_imgs_1.narrow(0, b, self.batch_size))
                    loss = self.criterion(output, noisy_imgs_2.narrow(0, b, self.batch_size))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


    def predict(self, test_input) -> torch.Tensor:
        """Generates a prediction (denoising) on the input

        Parameters
        ----------
        test_input: (N1, C, H, W), torch.tensor
            Tensor to be denoised by the network.

        Returns
        -------
        torch.tensor, (N1, C, H, W) 
            The denoised `test_input`
        """

        test_input = (test_input  / 255.0).float().to(self.device)
        # Seems reasonable to set model to eval mode and 
        # not compute gradients during predictions  
        self.model.eval()
        with torch.no_grad(): 
            return self.model(test_input) * 255.0
