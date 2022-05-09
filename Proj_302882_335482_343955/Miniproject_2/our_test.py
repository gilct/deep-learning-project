from model import *
import unittest
from alive_progress import alive_bar
from time import sleep

import torch
import torch.nn as nn
from torch import empty, randint, randn, rand
torch.set_grad_enabled(True)

SEED = 2022
torch.manual_seed(SEED)

from .others.test_helpers import *

class Testing(unittest.TestCase):

    # @unittest.skip("")
    def test_linear(self):
        # """Compare our implementation of a network 
        # with two linear layers with equivalent torch model."""

        # Define dimensions of data and linear layers 
        # (1 input, 1 hidden, 1 output)
        in_dim, hidden_dim, out_dim = 50, 20, 15
        n_samples = 200

        # Parameters of distribution of inputs and targets
        mean, std = 0, 20
        unif_lower, unif_upper = 10, 15
        train_input = empty(n_samples, in_dim) \
                        .normal_(mean, std)
        train_targets = empty(n_samples, out_dim) \
                        .uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same 
        # initial value of all weights (for reproducibility)
        init_val = 0.05
        model_no_torch = Sequential(Linear(in_dim, 
                                           hidden_dim, 
                                           init_val=init_val),
                                    Sigmoid(),
                                    Linear(hidden_dim, 
                                           out_dim, 
                                           init_val=init_val))

        model_torch = LinearTorchTest(in_dim, hidden_dim, out_dim)
        model_torch.apply(init_weights_wrapper(init_val=init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 10, 20

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        loss_places = 4
        stats_mean_placess = 5 
        stats_std_places = 6

        train_and_assert(self, nb_epochs, batch_size, 
                         train_input, train_targets,
                         model_no_torch, model_torch, 
                         criterion_no_torch, criterion_torch, 
                         optimizer_no_torch, optimizer_torch,
                         loss_places, stats_mean_placess,
                         stats_std_places)

    # @unittest.skip("")
    def test_linear_several(self):
        print()
        # """Test linear several"""
        # """Compare several of our implementations of a 
        # network with two linear layers with equivalent torch model."""

        tests_to_run = 40
        gen = torch.Generator()

        sucess_count = 0

        with alive_bar(tests_to_run, ctrl_c=True, title='Test progress', 
                       stats=False, dual_line=True, bar="squares") as bar:
            for i in range(tests_to_run):

                gen.manual_seed(SEED+i)

                # Define dimensions of data and linear layers (1 input, 1 hidden, 1 output)
                in_dim = randint(10, 50, (1,), generator=gen).item()
                hidden_dim = randint(5, 10, (1,), generator=gen).item()
                out_dim = randint(1, 5, (1,), generator=gen).item()
                n_samples = randint(20, 200, (1,), generator=gen).item()

                # Parameters of distribution of inputs and targets
                mean, std = 0, 20
                unif_lower, unif_upper = -20, 20
                train_input = empty(n_samples, in_dim) \
                                .normal_(mean, std, generator=gen)
                train_targets = empty(n_samples, out_dim) \
                                .uniform_(unif_lower,unif_upper, generator=gen)

                # Define our model and torch model with same 
                # initial value of all weights (for reproducibility)
                init_val = rand(1, generator=gen).item()

                model_no_torch = Sequential(Linear(in_dim, hidden_dim, init_val=init_val),
                                            Sigmoid(),
                                            Linear(hidden_dim, out_dim, init_val=init_val))

                model_torch = LinearTorchTest(in_dim, hidden_dim, out_dim)
                model_torch.apply(init_weights_wrapper(init_val=init_val))

                # Training parameters and variables
                lr = rand(1, generator=gen).item()
                nb_epochs = randint(1, 10, (1,), generator=gen).item()
                batch_size = randint(20,30, (1,), generator=gen).item()

                optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
                criterion_no_torch = MSE()

                optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
                criterion_torch = nn.MSELoss()

                # Standardize data
                mu, std = train_input.mean(), train_input.std()
                train_input.sub_(mu).div_(std)

                bar_text = f'-> Using seed {SEED+i}, sucess: {sucess_count}/{i}'
                bar.text = bar_text

                failed_once = False

                loss_places = 2
                stats_mean_places = 3
                stats_std_places = 3

                failed_once = train_and_assert(self, nb_epochs, batch_size, 
                                               train_input, train_targets,
                                               model_no_torch, model_torch, 
                                               criterion_no_torch, criterion_torch, 
                                               optimizer_no_torch, optimizer_torch,
                                               loss_places, stats_mean_places,
                                               stats_std_places, with_try=True)

                if not failed_once:
                    sucess_count += 1

                sleep(0.2)

                bar()

    # @unittest.skip("")
    def test_convolution_only(self):
        # """Test convolution"""
        # """Compare our implementation of a network 
        # with a convolution with equivalent torch model."""

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
        train_input = empty(n_samples, in_channels, height, width) \
                        .normal_(mean, std)
        train_targets = empty(n_samples, *out_dim) \
                        .uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial value of all weights (for reproducibility)
        init_val = 0.005
        model_no_torch = Sequential(Conv2d(in_channels=in_channels,
                                           out_channels=out_channels, 
                                           kernel_size=kernel_size, 
                                           stride=stride, 
                                           init_val=init_val))

        model_torch = ConvolutionTorchTestSmall(in_channels, 
                                                out_channels, 
                                                kernel_size, 
                                                stride)
        model_torch.apply(init_weights_wrapper(init_val=init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 4, 100

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        loss_places = 4
        stats_mean_places = 5 
        stats_std_places = 6

        train_and_assert(self, nb_epochs, batch_size, 
                         train_input, train_targets,
                         model_no_torch, model_torch, 
                         criterion_no_torch, criterion_torch, 
                         optimizer_no_torch, optimizer_torch,
                         loss_places, stats_mean_places,
                         stats_std_places)

    # @unittest.skip("")
    def test_convolution_linear_small(self):
        # """Test convolution + linear small"""
        # """Compare our implementation of a network 
        # with a convolution and a linear layer
        # with equivalent torch model.
        # """

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
        train_input = empty(n_samples, in_channels, height, width) \
                            .normal_(mean, std)
        train_targets = empty(n_samples, out_dim) \
                            .uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial value of all weights (for reproducibility)
        init_val = 0.005
        model_no_torch = Sequential(Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           init_val=init_val),
                                    ReLU(),
                                    Linear(hidden_dim,
                                           out_dim, 
                                           init_val=init_val))

        model_torch = ConvolutionLinearTorchTestSmall(in_channels, 
                                                      out_channels, 
                                                      kernel_size, 
                                                      stride, 
                                                      hidden_dim, 
                                                      out_dim)
        model_torch.apply(init_weights_wrapper(init_val=init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 10, 60

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        loss_places = 3
        stats_mean_places = 4
        stats_std_places = 6    

        train_and_assert(self, nb_epochs, batch_size, 
                         train_input, train_targets,
                         model_no_torch, model_torch, 
                         criterion_no_torch, criterion_torch, 
                         optimizer_no_torch, optimizer_torch,
                         loss_places, stats_mean_places,
                         stats_std_places)

    # @unittest.skip("")
    def test_convolution_linear_big(self):
        # """Test convolution + linear big"""
        # """Compare our implementation of a network 
        # with two convolutions and two linear layers
        # with equivalent torch model.
        # """

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
        train_input = empty(n_samples, in_channels, height, width) \
                        .normal_(mean, std)
        train_targets = empty(n_samples, out_dim) \
                        .uniform_(unif_lower,unif_upper)

        # Define our model and torch model with same initial 
        # value of all weights (for reproducibility)
        init_val = 0.005
        model_no_torch = Sequential(Conv2d(in_channels=in_channels, 
                                           out_channels=out_channels_1, 
                                           kernel_size=kernel_size_1, 
                                           stride=stride_1, 
                                           init_val=init_val),
                                    ReLU(),
                                    Conv2d(in_channels=out_channels_1, 
                                           out_channels=out_channels_2, 
                                           kernel_size=kernel_size_2, 
                                           stride=stride_2, 
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
        model_torch.apply(init_weights_wrapper(init_val=init_val))

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 9, 47

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        loss_places = 6
        stats_mean_places = 8
        stats_std_places = 7

        train_and_assert(self, nb_epochs, batch_size, 
                         train_input, train_targets,
                         model_no_torch, model_torch, 
                         criterion_no_torch, criterion_torch, 
                         optimizer_no_torch, optimizer_torch,
                         loss_places, stats_mean_places,
                         stats_std_places)

    # @unittest.skip("")
    def test_compute_conv_dim(self):
        print()
        # """Test convolution dimensions"""
        # Test convolution output shape corresponds to equivalent 
        # torch implementation as well as equality of the resulting 
        # output data itself

        tests_to_run = 100
        gen = torch.Generator()

        # Count of successful torch convolutions and our convolutions
        # Make sure both are equal at the end
        torch_count = 0
        my_count = 0

        with alive_bar(tests_to_run, ctrl_c=True, dual_line=True, 
                       stats=False, title='Test progress', bar="squares") as bar:
            for i in range(tests_to_run):
                gen.manual_seed(SEED+i)

                batch_size = randint(1, 10, (1,), generator=gen).item()
                in_channels, out_channels = randint(1, 10, (2,), generator=gen).tolist()
                kernel_size = tuple(randint(1, 30, (2,), generator=gen).tolist())
                stride = tuple(randint(1, 15, (2,), generator=gen).tolist())
                padding = tuple(randint(1, 30, (2,), generator=gen).tolist())
                dilation = tuple(randint(1, 10, (2,), generator=gen).tolist())
                input_h, input_w = randint(20, 700, (2,), generator=gen).tolist()
                input = randn(batch_size, in_channels, input_h, input_w, generator=gen)
                init_val = rand(1, generator=gen).item()

                bar_text = f'-> Using seed {SEED+i},'
                bar_text += f' --> {torch_count}/{i} Compatible shapes'
                bar.text = bar_text

                torch_conv = torch.nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size,
                                             stride=stride, 
                                             padding=padding,
                                             dilation=dilation)
                torch_conv.apply(init_weights_wrapper(init_val=init_val))

                no_torch_conv = Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       init_val=init_val)

                # Can happen that kernel size / other params simply 
                # arent't compatible with data shape
                try:
                    convolved_torch = torch_conv(input)
                    torch_count += 1
                    # bar.text = bar_text + f' --> Compatible shapes'
                    convolved_no_torch = no_torch_conv.forward(input)
                    my_count += 1 # should always be reached if torch conv is successful
                    self.assertEqual(convolved_torch.shape, convolved_no_torch.shape)
                    self.assertTrue(torch.allclose(convolved_torch, convolved_no_torch, rtol=1e-03, atol=1e-04))
                except RuntimeError:
                    # bar.text = bar_text + f' --> Incompatible shapes'
                    sleep(0.1)

                bar()

        # Make sure we have equal number of torch convs as our convs
        self.assertEqual(torch_count, my_count)
        
    # @unittest.skip("")
    def test_transpose_conv(self):

        tests_to_run = 500
        gen = torch.Generator()

        # Count of successful torch (transpose)convolutions 
        # and our convolutions. Make sure both are equal at the end
        torch_count = 0
        my_count = 0
        torch_t_count = 0
        my_t_count = 0

        with alive_bar(tests_to_run, ctrl_c=True, dual_line=True, 
                       stats=False, title='Test progress', bar="squares") as bar:
            for i in range(tests_to_run):
                gen.manual_seed(SEED+i)

                batch_size = randint(1, 3, (1,), generator=gen).item()
                in_ch_conv1, out_ch_conv1, out_ch_conv2, out_ch_t_conv1, out_ch_t_conv2 = \
                    randint(1, 10, (5,), generator=gen).tolist()

                k_size_conv1 = tuple(randint(1, 9, (2,), generator=gen).tolist())
                k_size_conv2 = tuple(randint(1, 10, (2,), generator=gen).tolist())
                k_size_t_conv1 = tuple(randint(1, 11, (2,), generator=gen).tolist())
                k_size_t_conv2 = tuple(randint(1, 7, (2,), generator=gen).tolist())
                    
                stride_conv1 = tuple(randint(1, 5, (2,), generator=gen).tolist())
                stride_conv2 = tuple(randint(1, 4, (2,), generator=gen).tolist())
                stride_t_conv1 = tuple(randint(1, 12, (2,), generator=gen).tolist())
                stride_t_conv2 = tuple(randint(1, 9, (2,), generator=gen).tolist())

                padding_conv1 = tuple(randint(0, 3, (2,), generator=gen).tolist())
                padding_conv2 = tuple(randint(0, 2, (2,), generator=gen).tolist())
                padding_t_conv1 = tuple(randint(0, 5, (2,), generator=gen).tolist())
                padding_t_conv2 = tuple(randint(0, 4, (2,), generator=gen).tolist())

                dilation_conv1 = tuple(randint(1, 4, (2,), generator=gen).tolist())
                dilation_conv2 = tuple(randint(1, 3, (2,), generator=gen).tolist())
                dilation_t_conv1 = tuple(randint(1, 6, (2,), generator=gen).tolist())
                dilation_t_conv2 = tuple(randint(1, 7, (2,), generator=gen).tolist())
                
                input_h, input_w = randint(20, 100, (2,), generator=gen).tolist()
                data = randn(batch_size, in_ch_conv1, input_h, input_w, generator=gen)
                init_val = rand(1, generator=gen).item()

                bar_text = f'-> Using seed {SEED+i},'
                bar_text += f' --> {torch_count}/{i} Compatible shapes'
                bar.text = bar_text

                torch_conv1 = nn.Conv2d(in_channels=in_ch_conv1, 
                                        out_channels=out_ch_conv1, 
                                        kernel_size=k_size_conv1, 
                                        stride=stride_conv1,
                                        padding=padding_conv1,
                                        dilation=dilation_conv1)
                torch_conv2 = nn.Conv2d(in_channels=out_ch_conv1, 
                                        out_channels=out_ch_conv2, 
                                        kernel_size=k_size_conv2, 
                                        stride=stride_conv2, 
                                        padding=padding_conv2,
                                        dilation=dilation_conv2)
                torch_t_conv1 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv2, 
                                                         out_channels=out_ch_t_conv1, 
                                                         kernel_size=k_size_t_conv1, 
                                                         stride=stride_t_conv1, 
                                                         dilation=dilation_t_conv1,
                                                         padding=padding_t_conv1)
                torch_t_conv2 = torch.nn.ConvTranspose2d(in_channels=out_ch_t_conv1, 
                                                         out_channels=out_ch_t_conv2, 
                                                         kernel_size=k_size_t_conv2, 
                                                         stride=stride_t_conv2, 
                                                         dilation=dilation_t_conv2,
                                                         padding=padding_t_conv2)

                torch_conv1.apply(init_weights_wrapper(init_val=init_val))
                torch_conv2.apply(init_weights_wrapper(init_val=init_val))
                torch_t_conv1.apply(init_weights_wrapper(init_val=init_val))
                torch_t_conv2.apply(init_weights_wrapper(init_val=init_val))

                no_torch_conv1 = Conv2d(in_channels=in_ch_conv1, 
                                        out_channels=out_ch_conv1, 
                                        kernel_size=k_size_conv1, 
                                        stride=stride_conv1,
                                        padding=padding_conv1,
                                        dilation=dilation_conv1,
                                        init_val=init_val)
                no_torch_conv2 = Conv2d(in_channels=out_ch_conv1, 
                                        out_channels=out_ch_conv2, 
                                        kernel_size=k_size_conv2, 
                                        stride=stride_conv2, 
                                        padding=padding_conv2,
                                        dilation=dilation_conv2,
                                        init_val=init_val)
                no_torch_t_conv1 = TransposeConv2d(in_channels=out_ch_conv2, 
                                                   out_channels=out_ch_t_conv1, 
                                                   kernel_size=k_size_t_conv1, 
                                                   stride=stride_t_conv1, 
                                                   dilation=dilation_t_conv1,
                                                   padding=padding_t_conv1,
                                                   init_val=init_val)
                no_torch_t_conv2 = TransposeConv2d(in_channels=out_ch_t_conv1, 
                                                   out_channels=out_ch_t_conv2, 
                                                   kernel_size=k_size_t_conv2, 
                                                   stride=stride_t_conv2, 
                                                   dilation=dilation_t_conv2,
                                                   padding=padding_t_conv2,
                                                   init_val=init_val)                           
                
                try:
                    torch_convolved_1 = torch_conv1(data)
                    torch_convolved_2 = torch_conv2(torch_convolved_1)
                    torch_count += 1

                    no_torch_convolved_1 = no_torch_conv1.forward(data)
                    no_torch_convolved_2 = no_torch_conv2.forward(no_torch_convolved_1)
                    my_count += 1

                    torch_t_convolved_1 = torch_t_conv1(torch_convolved_2)
                    torch_t_convolved_2 = torch_t_conv2(torch_t_convolved_1)
                    torch_t_count += 1

                    no_torch_t_convolved_1 = no_torch_t_conv1.forward(torch_convolved_2)
                    no_torch_t_convolved_2 = no_torch_t_conv2.forward(no_torch_t_convolved_1)
                    my_t_count += 1

                    self.assertEqual(torch_convolved_1.shape, no_torch_convolved_1.shape)
                    self.assertEqual(torch_convolved_2.shape, no_torch_convolved_2.shape)

                    self.assertEqual(torch_t_convolved_1.shape, no_torch_t_convolved_1.shape)
                    self.assertEqual(torch_t_convolved_2.shape, no_torch_t_convolved_2.shape)

                    self.assertTrue(torch.allclose(torch_t_convolved_1, no_torch_t_convolved_1, rtol=1e-4,atol=1e-2))
                    self.assertTrue(torch.allclose(torch_t_convolved_2, no_torch_t_convolved_2, rtol=1e-1,atol=1e-3))
                except RuntimeError as e:
                    pass

                bar()
        
        self.assertEqual(torch_count, my_count,
                         msg="Equal number of successful shapes for convolution")
        self.assertEqual(torch_t_count, my_t_count,
                         msg="Equal number of successful shapes for transpose convolution")

        print(f'{my_count}/{tests_to_run} runs had compatible convolution shapes')
        print(f'{my_t_count}/{tests_to_run} runs had compatible transpose convolution shapes')
    
    # @unittest.skip("")
    def test_simple_transpose_conv(self):
        # """Tests a simple double convolution followed 
        # by double transpose convolution with same kernels."""
        n_samples, channels, h, w = 20, 3, 32, 48
        out_ch_conv1, out_ch_conv2 = 32, 64
        kernel_size = (2,2)
        stride = 2
        data = torch.randn(n_samples, channels, h, w)
        init_val = 0.05

        torch_conv1 = nn.Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size, 
                                stride=stride)
        torch_conv2 = nn.Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size, 
                                stride=stride)
        torch_t_conv1 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv2, 
                                                 out_channels=out_ch_conv1, 
                                                 kernel_size=kernel_size, 
                                                 stride=stride)
        torch_t_conv2 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv1, 
                                                 out_channels=channels, 
                                                 kernel_size=kernel_size, 
                                                 stride=stride)

        torch_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_conv2.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv2.apply(init_weights_wrapper(init_val=init_val))

        no_torch_conv1 = Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                init_val=init_val)
        no_torch_conv2 = Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                init_val=init_val)
        no_torch_t_conv1 = TransposeConv2d(in_channels=out_ch_conv2, 
                                          out_channels=out_ch_conv1, 
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          init_val=init_val)
        no_torch_t_conv2 = TransposeConv2d(in_channels=out_ch_conv1, 
                                          out_channels=channels, 
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          init_val=init_val) 

    
        torch_convolved_1 = torch_conv1(data)
        torch_convolved_2 = torch_conv2(torch_convolved_1)

        no_torch_convolved_1 = no_torch_conv1.forward(data)
        no_torch_convolved_2 = no_torch_conv2.forward(no_torch_convolved_1)

        torch_t_convolved_1 = torch_t_conv1(torch_convolved_2)
        torch_t_convolved_2 = torch_t_conv2(torch_t_convolved_1)

        no_torch_t_convolved_1 = no_torch_t_conv1.forward(no_torch_convolved_2)
        no_torch_t_convolved_2 = no_torch_t_conv2.forward(no_torch_t_convolved_1)

        self.assertEqual(torch_convolved_1.shape, no_torch_convolved_1.shape)
        self.assertEqual(torch_convolved_2.shape, no_torch_convolved_2.shape)

        self.assertEqual(torch_t_convolved_1.shape, no_torch_t_convolved_1.shape)
        self.assertEqual(torch_t_convolved_2.shape, no_torch_t_convolved_2.shape)

        self.assertTrue(torch.allclose(torch_convolved_1, no_torch_convolved_1, rtol=1e-6,atol=1e-7))
        self.assertTrue(torch.allclose(torch_convolved_2, no_torch_convolved_2, rtol=1e-6,atol=1e-6))

        self.assertTrue(torch.allclose(torch_t_convolved_1, no_torch_t_convolved_1, rtol=1e-5,atol=1e-6))
        self.assertTrue(torch.allclose(torch_t_convolved_2, no_torch_t_convolved_2, rtol=1e-4,atol=1e-6))

        if False:
            print()
            print(f'Data : {list(data.shape)}')
            print(f' Torch    --> {list(torch_convolved_1.shape)}')
            print(f' No Torch --> {list(no_torch_convolved_1.shape)}')
            print(f'    Torch    --> {list(torch_convolved_2.shape)}')
            print(f'    No Torch --> {list(no_torch_convolved_2.shape)}')
            print(f'        Torch    --> {list(torch_t_convolved_1.shape)}')
            print(f'        No Torch --> {list(no_torch_t_convolved_1.shape)}')
            print(f'            Torch    --> {list(torch_t_convolved_2.shape)}')
            print(f'            No Torch --> {list(no_torch_t_convolved_2.shape)}')

    # @unittest.skip("")
    def test_transpose_conv_train(self):
        n_samples, channels, h, w = 20, 1, 32, 48
        out_ch_conv1, out_ch_conv2 = 32, 64
        kernel_size = (2,2)
        stride = 2
        torch.manual_seed(2022)
        train_input = torch.randn(n_samples, channels, h, w)

        # train_input = torch.randint(0,2,(n_samples, channels, h, w)).type(torch.FloatTensor)
        train_targets = train_input + 0.5
        init_val = 0.05

        torch_conv1 = nn.Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size, 
                                stride=stride)
        torch_conv2 = nn.Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size, 
                                stride=stride)
        torch_t_conv1 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv2, 
                                                out_channels=out_ch_conv1, 
                                                kernel_size=kernel_size, 
                                                stride=stride)
        torch_t_conv2 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv1, 
                                                out_channels=channels, 
                                                kernel_size=kernel_size, 
                                                stride=stride)
        torch_relu = nn.ReLU()

        torch_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_conv2.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv2.apply(init_weights_wrapper(init_val=init_val))

        no_torch_conv1 = Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                init_val=init_val)
        no_torch_conv2 = Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                init_val=init_val)
        no_torch_t_conv1 = TransposeConv2d(in_channels=out_ch_conv2, 
                                        out_channels=out_ch_conv1, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        init_val=init_val)
        no_torch_t_conv2 = TransposeConv2d(in_channels=out_ch_conv1, 
                                        out_channels=channels, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        init_val=init_val) 
        no_torch_relu_1 = ReLU()
        no_torch_relu_2 = ReLU()

        model_no_torch = Sequential(no_torch_conv1, no_torch_conv2, no_torch_relu_1, no_torch_t_conv1, no_torch_relu_2, no_torch_t_conv2)
        model_torch = nn.Sequential(torch_conv1, torch_conv2, torch_relu, torch_t_conv1, torch_relu, torch_t_conv2)

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 10, 5

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        loss_places = 6
        stats_mean_placess = 6
        stats_std_places = 6

        train_and_assert(self, nb_epochs, batch_size, 
                         train_input, train_targets,
                         model_no_torch, model_torch, 
                         criterion_no_torch, criterion_torch, 
                         optimizer_no_torch, optimizer_torch,
                         loss_places, stats_mean_placess,
                         stats_std_places)
    
    # @unittest.skip("")
    def test_transpose_conv_train_diff_shapes(self):

        n_samples, channels, h, w = 20, 3, 32, 48
        out_ch_conv1, out_ch_conv2 = 32, 64
        kernel_size = (4,2)
        stride = (2,3)
        torch.manual_seed(2022)
        train_input = torch.randn(n_samples, channels, h, w)
        train_targets = torch.randn(n_samples, channels, 21, 45) # pre-determined to be output shape  
        init_val = 0.01
        dilation = (2,2)
        padding = (3,0)

        torch_conv1 = nn.Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size, 
                                stride=stride)
        torch_conv2 = nn.Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size, 
                                stride=stride)
        torch_t_conv1 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv2, 
                                                out_channels=out_ch_conv1, 
                                                kernel_size=kernel_size, 
                                                stride=stride,
                                                dilation=dilation,
                                                padding=padding)
        torch_t_conv2 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv1, 
                                                out_channels=channels, 
                                                kernel_size=kernel_size, 
                                                stride=stride,
                                                dilation=dilation,
                                                padding=padding)
        torch_relu = nn.ReLU()

        torch_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_conv2.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv2.apply(init_weights_wrapper(init_val=init_val))

        no_torch_conv1 = Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                init_val=init_val)
        no_torch_conv2 = Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size, 
                                stride=stride,
                                init_val=init_val)
        no_torch_t_conv1 = TransposeConv2d(in_channels=out_ch_conv2, 
                                        out_channels=out_ch_conv1, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        dilation=dilation,
                                        padding=padding,
                                        init_val=init_val)
        no_torch_t_conv2 = TransposeConv2d(in_channels=out_ch_conv1, 
                                        out_channels=channels, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        dilation=dilation,
                                        padding=padding,
                                        init_val=init_val)
        no_torch_relu_1 = ReLU()
        no_torch_relu_2 = ReLU()

        model_no_torch = Sequential(no_torch_conv1, no_torch_conv2, 
                                    no_torch_relu_1, no_torch_t_conv1, 
                                    no_torch_relu_2, no_torch_t_conv2)
        model_torch = nn.Sequential(torch_conv1, torch_conv2, 
                                    torch_relu, torch_t_conv1, 
                                    torch_relu, torch_t_conv2)

        # Training parameters and variables
        lr, nb_epochs, batch_size = 1e-1, 10, 5

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()

        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        loss_places = 6
        stats_mean_placess = 8
        stats_std_places = 7

        train_and_assert(self, nb_epochs, batch_size, 
                         train_input, train_targets,
                         model_no_torch, model_torch, 
                         criterion_no_torch, criterion_torch, 
                         optimizer_no_torch, optimizer_torch,
                         loss_places, stats_mean_placess,
                         stats_std_places)

    # @unittest.skip("")
    def test_transpose_conv_grad(self):

        n_samples, channels, h, w  = 200, 3, 41, 27
        out_ch_conv1, out_ch_conv2 = 32, 64


        kernel_size_1 = (3,2)
        kernel_size_2 = (4,1)
        kernel_size_t_1 = (1,5)
        kernel_size_t_2 = (2,3)

        stride_1 = (2,1)
        stride_2 = (4,2)    
        stride_t_1 = (2,1)
        stride_t_2 = (1,4)

        padding_1 = (0,3)
        padding_2 = (2,1)
        padding_t_1 = (1,4)
        padding_t_2 = (0,2)

        dilation_1 = (1,2)
        dilation_2 = (4,2)
        dilation_t_1 = (1,2)
        dilation_t_2 = (3,5)

        train_input = torch.randint(0,20,(n_samples, channels, h, w)).type(torch.FloatTensor)
        train_targets = torch.randint(-15,15,(n_samples, channels, 6, 71)).type(torch.FloatTensor)
        init_val = 0.05


        no_torch_conv1 = Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size_1, 
                                stride=stride_1,
                                padding=padding_1,
                                dilation=dilation_1,
                                init_val=init_val)
        no_torch_conv2 = Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size_2, 
                                stride=stride_2,
                                padding=padding_2,
                                dilation=dilation_2,
                                init_val=init_val)
        no_torch_t_conv1 = TransposeConv2d(in_channels=out_ch_conv2, 
                                           out_channels=out_ch_conv1, 
                                           kernel_size=kernel_size_t_1, 
                                           stride=stride_t_1,   
                                           padding=padding_t_1,
                                           dilation=dilation_t_1,
                                           init_val=init_val)
        no_torch_t_conv2 = TransposeConv2d(in_channels=out_ch_conv1, 
                                           out_channels=channels, 
                                           kernel_size=kernel_size_t_2, 
                                           stride=stride_t_2,      
                                           padding=padding_t_2,
                                           dilation=dilation_t_2,
                                           init_val=init_val) 
        no_torch_relu_1 = ReLU()
        no_torch_relu_2 = ReLU()

        no_torch_sigmoid_1 = Sigmoid()
        no_torch_sigmoid_2 = Sigmoid()

        model_no_torch = Sequential(no_torch_conv1, 
                                    no_torch_sigmoid_1, 
                                    no_torch_conv2, 
                                    no_torch_relu_1, 
                                    no_torch_t_conv1, 
                                    no_torch_relu_2, 
                                    no_torch_t_conv2, 
                                    no_torch_sigmoid_2)

        torch_conv1 = nn.Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size_1, 
                                stride=stride_1,
                                padding=padding_1,
                                dilation=dilation_1)
        torch_conv2 = nn.Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size_2, 
                                stride=stride_2,
                                padding=padding_2,
                                dilation=dilation_2)
        torch_t_conv1 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv2, 
                                                 out_channels=out_ch_conv1, 
                                                 kernel_size=kernel_size_t_1, 
                                                 stride=stride_t_1,   
                                                 padding=padding_t_1,
                                                 dilation=dilation_t_1)
        torch_t_conv2 = torch.nn.ConvTranspose2d(in_channels=out_ch_conv1, 
                                                 out_channels=channels, 
                                                 kernel_size=kernel_size_t_2, 
                                                 stride=stride_t_2,      
                                                 padding=padding_t_2,
                                                 dilation=dilation_t_2)

        torch_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_conv2.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_t_conv2.apply(init_weights_wrapper(init_val=init_val))

        model_torch = GradNet(torch_conv1, torch_conv2, 
                              torch_t_conv1, torch_t_conv2)

        lr, nb_epochs, batch_size = 1e-1, 10, 20
        batches = math.ceil(n_samples / batch_size)

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()
                
        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        for e in range(nb_epochs):
            for b, (input, targets) in enumerate(zip(train_input.split(batch_size),
                                                   train_targets.split(batch_size))):

                output_no_torch = model_no_torch.forward(input)
                input.requires_grad = True
                output_torch = model_torch(input)

                loss_no_torch = criterion_no_torch.forward(output_no_torch, targets)
                loss_torch = criterion_torch(output_torch, targets)   

                optimizer_no_torch.zero_grad()
                optimizer_torch.zero_grad() 

                model_no_torch.backward(criterion_no_torch.backward())
                loss_torch.backward()

                self.assertTrue(
                    torch.allclose(model_torch.grads['sigmoid_2'],no_torch_sigmoid_2.grad), 
                    msg=f'Equal sigmoid_2 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['t_conv2'],no_torch_t_conv2.grad), 
                    msg=f'Equal transpose conv_2 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['relu_2'],no_torch_relu_2.grad), 
                    msg=f'Equal relu_2 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['t_conv1'],no_torch_t_conv1.grad), 
                    msg=f'Equal transpose conv_1 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['relu_1'],no_torch_relu_1.grad), 
                    msg=f'Equal relu_1 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['conv2'],no_torch_conv2.grad), 
                    msg=f'Equal conv_2 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['sigmoid_1'],no_torch_sigmoid_1.grad), 
                    msg=f'Equal sigmoid_1 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['conv1'],no_torch_conv1.grad), 
                    msg=f'Equal conv_1 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                    
                self.assertTrue(
                    torch.allclose(torch_conv1.bias.grad,no_torch_conv1.grad_bias),
                    msg=f'Equal conv_1 bias grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.bias.grad,no_torch_conv2.grad_bias, atol=1e-6),
                    msg=f'Equal conv_2 bias grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv1.weight.grad,no_torch_conv1.grad_weight),
                    msg=f'Equal conv_1 weight grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.weight.grad,no_torch_conv2.grad_weight),
                    msg=f'Equal conv_2 weight grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv1.bias,no_torch_conv1.bias),
                    msg=f'Equal conv_1 bias @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.bias,no_torch_conv2.bias),
                    msg=f'Equal conv_2 bias @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv1.weight,no_torch_conv1.weight),
                    msg=f'Equal conv_1 weight @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.weight,no_torch_conv2.weight),
                    msg=f'Equal conv_2 weight @: epoch {e}/{nb_epochs} - batch {b}/{batches}')

                self.assertTrue(
                    torch.allclose(torch_t_conv1.bias.grad,no_torch_t_conv1.conv.grad_bias),
                    msg=f'Equal transpose conv_1 bias grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_t_conv2.bias.grad,no_torch_t_conv2.conv.grad_bias),
                    msg=f'Equal transpose conv_2 bias grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_t_conv1.weight.grad,no_torch_t_conv1.conv.grad_weight.transpose(0,1).flip(2,3)),
                    msg=f'Equal transpose conv_1 weight grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_t_conv2.weight.grad,no_torch_t_conv2.conv.grad_weight.transpose(0,1).flip(2,3)),
                    msg=f'Equal transpose conv_2 weight grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_t_conv1.bias,no_torch_t_conv1.conv.bias),
                    msg=f'Equal transpose conv_1 bias @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_t_conv2.bias,no_torch_t_conv2.conv.bias),
                    msg=f'Equal transpose conv_2 bias @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_t_conv1.weight,no_torch_t_conv1.conv.weight.transpose(0,1).flip(2,3)),
                    msg=f'Equal transpose conv_1 weight @: epoch {e}/{nb_epochs} - batch {b}/{batches}')                 
                self.assertTrue(
                    torch.allclose(torch_t_conv2.weight,no_torch_t_conv2.conv.weight.transpose(0,1).flip(2,3)),
                    msg=f'Equal transpose conv_2 weight @: epoch {e}/{nb_epochs} - batch {b}/{batches}')

                optimizer_no_torch.step()
                optimizer_torch.step()

    # @unittest.skip("")
    def test_conv_grad(self):

        n_samples, channels, h, w = 200, 3, 190, 35
        out_ch_conv1, out_ch_conv2 = 3, 6

        kernel_size_1 = (3,2)
        kernel_size_2 = (4,1)

        stride_1 = (2,1)
        stride_2 = (4,2)

        padding_1 = (0,3)
        padding_2 = (2,1)

        dilation_1 = (1,1)
        dilation_2 = (1,2)

        torch.manual_seed(2022)
        train_input = torch.randint(0,20,(n_samples, channels, h, w)).type(torch.FloatTensor)
        train_targets = torch.randint(-15,15,(n_samples, out_ch_conv2, 24, 21)).type(torch.FloatTensor)
        init_val = 0.05

        torch_conv1 = nn.Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size_1, 
                                stride=stride_1,
                                padding=padding_1,
                                dilation=dilation_1)
        torch_conv2 = nn.Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size_2, 
                                stride=stride_2,
                                padding=padding_2,
                                dilation=dilation_2)
        torch_conv1.apply(init_weights_wrapper(init_val=init_val))
        torch_conv2.apply(init_weights_wrapper(init_val=init_val))

        no_torch_conv1 = Conv2d(in_channels=channels, 
                                out_channels=out_ch_conv1, 
                                kernel_size=kernel_size_1, 
                                stride=stride_1,
                                padding=padding_1,
                                dilation=dilation_1,
                                init_val=init_val)
        no_torch_conv2 = Conv2d(in_channels=out_ch_conv1, 
                                out_channels=out_ch_conv2, 
                                kernel_size=kernel_size_2, 
                                stride=stride_2,
                                padding=padding_2,
                                dilation=dilation_2,
                                init_val=init_val)
        no_torch_relu_1 = ReLU()

        model_no_torch = Sequential(no_torch_conv1, no_torch_relu_1, no_torch_conv2)
        model_torch = GradNetConvOnly(torch_conv1, torch_conv2)

        lr, nb_epochs, batch_size = 1e-1, 10, 5
        batches = math.ceil(n_samples / batch_size) 

        optimizer_no_torch = SGD(model_no_torch.param(), lr=lr)
        criterion_no_torch = MSE()

        optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion_torch = nn.MSELoss()
                
        # Standardize data
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)

        for e in range(nb_epochs):
            for b, (input, targets) in enumerate(zip(train_input.split(batch_size),
                                                   train_targets.split(batch_size))):
                
                output_no_torch = model_no_torch.forward(input)
                input.requires_grad = True
                output_torch = model_torch(input)

                loss_no_torch = criterion_no_torch.forward(output_no_torch, targets)
                loss_torch = criterion_torch(output_torch, targets)   

                optimizer_no_torch.zero_grad()
                optimizer_torch.zero_grad() 

                model_no_torch.backward(criterion_no_torch.backward())
                loss_torch.backward()

                self.assertTrue(
                    torch.allclose(model_torch.grads['relu_1'],no_torch_relu_1.grad), 
                    msg=f'Equal relu_1 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['conv2'],no_torch_conv2.grad), 
                    msg=f'Equal conv_2 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(model_torch.grads['conv1'],no_torch_conv1.grad), 
                    msg=f'Equal conv_1 grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv1.bias.grad,no_torch_conv1.grad_bias),
                    msg=f'Equal conv_1 bias grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.bias.grad,no_torch_conv2.grad_bias, atol=1e-6),
                    msg=f'Equal conv_2 bias grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv1.weight.grad,no_torch_conv1.grad_weight),
                    msg=f'Equal conv_1 weight grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.weight.grad,no_torch_conv2.grad_weight),
                    msg=f'Equal conv_2 weight grad @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv1.bias,no_torch_conv1.bias),
                    msg=f'Equal conv_1 bias @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.bias,no_torch_conv2.bias),
                    msg=f'Equal conv_2 bias @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv1.weight,no_torch_conv1.weight),
                    msg=f'Equal conv_1 weight @: epoch {e}/{nb_epochs} - batch {b}/{batches}')
                self.assertTrue(
                    torch.allclose(torch_conv2.weight,no_torch_conv2.weight),
                    msg=f'Equal conv_2 weight @: epoch {e}/{nb_epochs} - batch {b}/{batches}')

                optimizer_no_torch.step()
                optimizer_torch.step()

    
if __name__ == '__main__':
    unittest.main(verbosity=2)