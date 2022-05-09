import torch
import torch.nn as nn

class ConvRes(nn.Module):
    """
    ConvRes classifier with 8 hidden layers and residuals

    Attributes
    ----------
    self.conv1 : nn.Conv2d
    self.conv2 : nn.Conv2d
    self.conv3 : nn.Conv2d
    self.conv4 : nn.Conv2d
    self.conv5 : nn.Conv2d
    self.conv6 : nn.Conv2d
    self.conv7 : nn.Conv2d
    self.conv8 : nn.Conv2d

    Methods
    -------
    forward(self, x)
        Performs a forward pass on x
        
    """
    def __init__(self):
        super(ConvRes, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding = 1)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding = 1)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding = 1)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding = 1)
        self.conv8 = nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding = 1)


    def forward(self, x):
        """
            Performs a forward pass
        """
        input = x
        y = nn.LeakyReLU(inplace = True)(self.conv1(x))
        x = nn.LeakyReLU(inplace = True)(self.conv2(y))
        x = nn.LeakyReLU(inplace = True)(self.conv3(x))
        x = nn.LeakyReLU(inplace = True)(self.conv4(x))
        x = nn.LeakyReLU(inplace = True)(self.conv5(x))
        x = nn.LeakyReLU(inplace = True)(self.conv6(x))
        x = nn.LeakyReLU(inplace = True)(self.conv7(x))
        x = x + y
        x = self.conv8(x)
        x = x + input
        x = x.clamp(0, 1)
        return x    