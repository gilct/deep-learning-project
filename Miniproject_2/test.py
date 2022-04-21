from model import *
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import empty
torch.set_grad_enabled(True)

SEED = 2022
torch.manual_seed(SEED)

from others.others import compute_conv_output_shape

class LinearTorchTest(nn.Module):
    """Torch module for testing a network with two linear layers"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvolutionTorchTestSmall(nn.Module):
    """Torch module for testing convolution only"""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        x = self.conv1(x)
        return x

class ConvolutionLinearTorchTestSmall(nn.Module):
    """Torch module for testing convolution and a linear layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.hidden_dim)
        x = self.fc1(x)
        return x

class ConvolutionLinearTorchTestBig(nn.Module):
    """Torch module for testing two convolutions and two linear layers"""
    def __init__(self,in_channels,
                 out_channels_1, 
                 out_channels_2, 
                 kernel_size_1, 
                 kernel_size_2,
                 stride_1, 
                 stride_2,  
                 hidden_dim_1, 
                 hidden_dim_2,
                 out_dim):
        super().__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=kernel_size_1, stride=stride_1)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=kernel_size_2, stride=stride_2)
        self.fc1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc2 = nn.Linear(hidden_dim_2, out_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.hidden_dim_1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Testing(unittest.TestCase):
    def test_linear(self):
        """Compare our implementation of a network with two linear layers with equivalent torch model."""

        # Define dimensions of data and linear layers (1 input, 1 hidden, 1 output)
        in_dim, hidden_dim, out_dim = 50, 20, 15
        n_samples = 200

        # Parameters of distribution of inputs and targets
        mean, std = 0, 20
        unif_lower, unif_upper = 10, 15
        train_input = empty(n_samples, in_dim).normal_(mean, std)
        train_targets = empty(n_samples, out_dim).uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial value of all weights (for reproducibility)
        init_val = 0.05
        model_no_torch = Sequential(Linear(in_dim, hidden_dim, init_val=init_val),
                                    Sigmoid(),
                                    Linear(hidden_dim, out_dim, init_val=init_val))

        model_torch = LinearTorchTest(in_dim, hidden_dim, out_dim)
        with torch.no_grad():
            model_torch.fc1.weight = nn.Parameter(torch.full_like(model_torch.fc1.weight, init_val))
            model_torch.fc1.bias = nn.Parameter(torch.full_like(model_torch.fc1.bias, init_val))
            model_torch.fc2.weight = nn.Parameter(torch.full_like(model_torch.fc2.weight, init_val))
            model_torch.fc2.bias = nn.Parameter(torch.full_like(model_torch.fc2.bias, init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 10, 20

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        # Train
        for _ in range(nb_epochs):
            for input, targets in zip(train_input.split(batch_size),
                                      train_targets.split(batch_size)):

                output_no_torch = model_no_torch.forward(input)
                output_torch = model_torch(input)

                # Retain grad for comparing MSE gradients, see below
                # output_torch.retain_grad()

                # Store statistics of the output for comparison
                stats_no_torch = (output_no_torch.mean().item(), output_no_torch.std().item())
                stats_torch = (output_torch.mean().item(), output_torch.std().item())

                loss_no_torch = criterion_no_torch.forward(output_no_torch, targets)
                loss_torch = criterion_torch(output_torch, targets)   

                optimizer_no_torch.zero_grad()
                optimizer_torch.zero_grad() 

                model_no_torch.backward(criterion_no_torch.backward())
                loss_torch.backward()

                # Compare gradients of MSE (make sure .retain_grad() is called on output_torch)
                # print(criterion_no_torch.backward())
                # print(output_torch.grad)

                optimizer_no_torch.step()
                optimizer_torch.step()

                # Assertions
                self.assertAlmostEqual(loss_no_torch.item(), loss_torch.item(), places=4, msg="Equal losses")
                self.assertAlmostEqual(stats_no_torch[0], stats_torch[0], places=5, msg="Equal mean of preds")
                self.assertAlmostEqual(stats_no_torch[1], stats_torch[1], places=6, msg="Equal std of preds")

    def test_convolution_small(self):
        """Compare our implementation of a network with a convolution with equivalent torch model."""

        # Define dimensions of data and convolution layer
        in_channels, height, width = 3, 32, 32
        n_samples = 1200
        out_channels = 5
        out_dim = (out_channels, 3, 2)
        kernel_size = (2, 4)
        stride = 15

        # Parameters of distribution of inputs and targets
        mean, std = 0, 20
        unif_lower, unif_upper = 10, 15
        train_input = empty(n_samples, in_channels, height, width).normal_(mean, std)
        train_targets = empty(n_samples, *out_dim).uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial value of all weights (for reproducibility)
        init_val = 0.005
        model_no_torch = Sequential(Conv2d(in_channels = in_channels,
                                           out_channels = out_channels, 
                                           kernel_size = kernel_size, 
                                           stride = stride, 
                                           init_val=init_val))

        model_torch = ConvolutionTorchTestSmall(in_channels, out_channels, kernel_size, stride)
        with torch.no_grad():
            model_torch.conv1.weight = nn.Parameter(torch.full_like(model_torch.conv1.weight, init_val))
            model_torch.conv1.bias = nn.Parameter(torch.full_like(model_torch.conv1.bias, init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 4, 100

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        # Train
        for _ in range(nb_epochs):
            for input, targets in zip(train_input.split(batch_size),
                                      train_targets.split(batch_size)):

                output_no_torch = model_no_torch.forward(input)
                output_torch = model_torch(input)

                # Retain grad for comparing MSE gradients, see below
                # output_torch.retain_grad()

                # Store statistics of the output for comparison
                stats_no_torch = (output_no_torch.mean().item(), output_no_torch.std().item())
                stats_torch = (output_torch.mean().item(), output_torch.std().item())

                loss_no_torch = criterion_no_torch.forward(output_no_torch, targets)
                loss_torch = criterion_torch(output_torch, targets)   

                optimizer_no_torch.zero_grad()
                optimizer_torch.zero_grad() 

                model_no_torch.backward(criterion_no_torch.backward())
                loss_torch.backward()

                # Compare gradients of MSE (make sure .retain_grad() is called on output_torch)
                # print(criterion_no_torch.backward())
                # print(output_torch.grad)

                optimizer_no_torch.step()
                optimizer_torch.step()

                # Assertions
                self.assertAlmostEqual(loss_no_torch.item(), loss_torch.item(), places=4, msg="Equal losses")
                self.assertAlmostEqual(stats_no_torch[0], stats_torch[0], places=5, msg="Equal mean of preds")
                self.assertAlmostEqual(stats_no_torch[1], stats_torch[1], places=6, msg="Equal std of preds")

    def test_convolution_linear_small(self):
        """Compare our implementation of a network with a convolution and a linear layer
        with equivalent torch model.
        """

        # Define dimensions of data and convolution layer
        in_channels, height, width = 3, 32, 32
        n_samples = 120
        out_channels = 5
        hidden_dim, out_dim = 30, 8
        kernel_size = (2, 4)
        stride = 15

        # Parameters of distribution of inputs and targets
        mean, std = 0, 20
        unif_lower, unif_upper = 10, 15
        train_input = empty(n_samples, in_channels, height, width).normal_(mean, std)
        train_targets = empty(n_samples, out_dim).uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial value of all weights (for reproducibility)
        init_val = 0.005
        model_no_torch = Sequential(Conv2d(in_channels = in_channels,
                                           out_channels = out_channels,
                                           kernel_size = kernel_size,
                                           stride = stride,
                                           init_val=init_val),
                                    ReLU(),
                                    Linear(hidden_dim,
                                           out_dim, 
                                           init_val=init_val))

        model_torch = ConvolutionLinearTorchTestSmall(in_channels, out_channels, kernel_size, stride, hidden_dim, out_dim)
        with torch.no_grad():
            model_torch.conv1.weight = nn.Parameter(torch.full_like(model_torch.conv1.weight, init_val))
            model_torch.conv1.bias = nn.Parameter(torch.full_like(model_torch.conv1.bias, init_val))
            model_torch.fc1.weight = nn.Parameter(torch.full_like(model_torch.fc1.weight, init_val))
            model_torch.fc1.bias = nn.Parameter(torch.full_like(model_torch.fc1.bias, init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 10, 60

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        # Train
        for _ in range(nb_epochs):
            for input, targets in zip(train_input.split(batch_size),
                                      train_targets.split(batch_size)):

                output_no_torch = model_no_torch.forward(input)
                output_torch = model_torch(input)

                # Retain grad for comparing MSE gradients, see below
                # output_torch.retain_grad()

                # Store statistics of the output for comparison
                stats_no_torch = (output_no_torch.mean().item(), output_no_torch.std().item())
                stats_torch = (output_torch.mean().item(), output_torch.std().item())

                loss_no_torch = criterion_no_torch.forward(output_no_torch, targets)
                loss_torch = criterion_torch(output_torch, targets)   

                optimizer_no_torch.zero_grad()
                optimizer_torch.zero_grad() 

                model_no_torch.backward(criterion_no_torch.backward())
                loss_torch.backward()

                # Compare gradients of MSE (make sure .retain_grad() is called on output_torch)
                # print(criterion_no_torch.backward())
                # print(output_torch.grad)

                optimizer_no_torch.step()
                optimizer_torch.step()

                # Assertions
                self.assertAlmostEqual(loss_no_torch.item(), loss_torch.item(), places=2, msg="Equal losses")
                self.assertAlmostEqual(stats_no_torch[0], stats_torch[0], places=3, msg="Equal mean of preds")
                self.assertAlmostEqual(stats_no_torch[1], stats_torch[1], places=4, msg="Equal std of preds")

    def test_convolution_linear_big(self):
        """Compare our implementation of a network with two convolutions and two linear layers
        with equivalent torch model.
        """

        # Define dimensions of data and convolution layer
        in_channels, height, width = 3, 32, 32
        n_samples = 237
        hidden_dim_1, hidden_dim_2, out_dim = 3, 2, 1
        out_channels_1, out_channels_2 = 5, 3
        kernel_size_1, kernel_size_2 = (2, 4), (2, 4)
        stride_1, stride_2 = 3, 15

        # Parameters of distribution of inputs and targets
        mean, std = 0, 20
        unif_lower, unif_upper = 10, 15
        train_input = empty(n_samples, in_channels, height, width).normal_(mean, std)
        train_targets = empty(n_samples, out_dim).uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial value of all weights (for reproducibility)
        init_val = 0.005
        model_no_torch = Sequential(Conv2d(in_channels = in_channels, 
                                           out_channels = out_channels_1, 
                                           kernel_size = kernel_size_1, 
                                           stride = stride_1, 
                                           init_val=init_val),
                                    ReLU(),
                                    Conv2d(in_channels = out_channels_1, 
                                           out_channels = out_channels_2, 
                                           kernel_size = kernel_size_2, 
                                           stride = stride_2, 
                                           init_val=init_val),
                                    ReLU(),
                                    Linear(hidden_dim_1,
                                           hidden_dim_2,
                                           init_val=init_val),
                                    ReLU(),
                                    Linear(hidden_dim_2,
                                           out_dim, 
                                           init_val=init_val),
                                    ReLU())

        model_torch = ConvolutionLinearTorchTestBig(in_channels,
                                                    out_channels_1, 
                                                    out_channels_2, 
                                                    kernel_size_1, 
                                                    kernel_size_2,
                                                    stride_1, 
                                                    stride_2,
                                                    hidden_dim_1,
                                                    hidden_dim_2,
                                                    out_dim)
        with torch.no_grad():
            model_torch.fc1.weight = nn.Parameter(torch.full_like(model_torch.fc1.weight, init_val))
            model_torch.fc1.bias = nn.Parameter(torch.full_like(model_torch.fc1.bias, init_val))
            model_torch.fc2.weight = nn.Parameter(torch.full_like(model_torch.fc2.weight, init_val))
            model_torch.fc2.bias = nn.Parameter(torch.full_like(model_torch.fc2.bias, init_val))
            model_torch.conv1.weight = nn.Parameter(torch.full_like(model_torch.conv1.weight, init_val))
            model_torch.conv1.bias = nn.Parameter(torch.full_like(model_torch.conv1.bias, init_val))
            model_torch.conv2.weight = nn.Parameter(torch.full_like(model_torch.conv2.weight, init_val))
            model_torch.conv2.bias = nn.Parameter(torch.full_like(model_torch.conv2.bias, init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 9, 47

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        # Train
        for _ in range(nb_epochs):
            for input, targets in zip(train_input.split(batch_size),
                                      train_targets.split(batch_size)):

                output_no_torch = model_no_torch.forward(input)
                output_torch = model_torch(input)

                # Retain grad for comparing MSE gradients, see below
                # output_torch.retain_grad()

                # Store statistics of the output for comparison
                stats_no_torch = (output_no_torch.mean().item(), output_no_torch.std().item())
                stats_torch = (output_torch.mean().item(), output_torch.std().item())

                loss_no_torch = criterion_no_torch.forward(output_no_torch, targets)
                loss_torch = criterion_torch(output_torch, targets)   

                optimizer_no_torch.zero_grad()
                optimizer_torch.zero_grad() 

                model_no_torch.backward(criterion_no_torch.backward())
                loss_torch.backward()

                # Compare gradients of MSE (make sure .retain_grad() is called on output_torch)
                # print(criterion_no_torch.backward())
                # print(output_torch.grad)

                optimizer_no_torch.step()
                optimizer_torch.step()

                # Assertions
                self.assertAlmostEqual(loss_no_torch.item(), loss_torch.item(), places=2, msg="Equal losses")
                self.assertAlmostEqual(stats_no_torch[0], stats_torch[0], places=2, msg="Equal mean of preds")
                self.assertAlmostEqual(stats_no_torch[1], stats_torch[1], places=2, msg="Equal std of preds")

    def test_compute_conv_dim(self):
        """Test the computation of the convolution dimensions"""
        in_channels, out_channels = 3, 5
        kernel_size = (21, 43)
        stride = (7,13)
        padding = (33,2)
        dilation = (14,11)
        input_h, input_w = 351, 2248

        conv = torch.nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, 
                               kernel_size=kernel_size,
                               stride=stride, 
                               padding=padding,
                               dilation=dilation)
        x = torch.randn((4, in_channels, input_h, input_w))
        conv_res = conv(x)
        conv_res_h, conv_res_w = conv_res.size(2), conv_res.size(3)

        h, w = compute_conv_output_shape((input_h, input_w),
                                   kernel_size,
                                   stride,
                                   padding,
                                   dilation)
        self.assertEqual((conv_res_h, conv_res_w), (h, w))

if __name__ == '__main__':
    unittest.main()