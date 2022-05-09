import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Functions -------------------

def init_weights_wrapper(init_val=0.05):
    """Wrapper function for init_weights() in order
    to accept the `init_val` as an argument
    
    Parameters
    ----------
    init_val : float, optional
       Default value of all weights (default is 0.05)

    Returns
    -------
    function
        The init_weights function
    """
    @torch.no_grad()
    def init_weights(module):
        """Initializes the weights the module it is applied to
        if it is one of nn.Linear, nn.Conv2d or nn.ConvTranspose2d
        
        Parameters
        ----------
        module : nn.Module
            The module to initialize the weights of
        """
        if isinstance(module, (nn.Linear, 
                               nn.Conv2d, 
                               nn.ConvTranspose2d)):
            module.weight.fill_(init_val)
            module.bias.fill_(init_val)
    return init_weights

def train_and_assert(self_test, nb_epochs, batch_size, 
                     train_input, train_targets,
                     model_no_torch, model_torch, 
                     criterion_no_torch, criterion_torch, 
                     optimizer_no_torch, optimizer_torch,
                     loss_places, stats_mean_places, 
                     stats_std_places, with_try=False):
    """Run training with assertions for tests"""

    failed_once = False

    for _ in range(nb_epochs):
        for input, targets in zip(train_input.split(batch_size),
                                    train_targets.split(batch_size)):

            output_no_torch = model_no_torch.forward(input)
            output_torch = model_torch(input)

            # Retain grad for comparing MSE gradients, see below
            # output_torch.retain_grad()

            # Store statistics (mean and std) of the output for comparison
            stats_no_torch = (output_no_torch.mean().item(), 
                              output_no_torch.std().item())
            stats_torch = (output_torch.mean().item(), 
                           output_torch.std().item())

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
            try:
                self_test.assertAlmostEqual(loss_no_torch.item(), 
                                            loss_torch.item(), 
                                            places=loss_places, 
                                            msg="Equal losses")
                self_test.assertAlmostEqual(stats_no_torch[0], 
                                            stats_torch[0],
                                            places=stats_mean_places, 
                                            msg="Equal mean of preds")
                self_test.assertAlmostEqual(stats_no_torch[1], 
                                            stats_torch[1], 
                                            places=stats_std_places, 
                                            msg="Equal std of preds")
            except AssertionError as a:
                if not with_try:
                    raise a
                # If loss > 1000 we don't can expect the losses to be close because it's likely
                # diverging to + inf
                elif abs(loss_no_torch.item()) > 10e2 or torch.isnan(loss_no_torch) or torch.isnan(loss_torch):
                    failed_once = True
                    pass
                else:
                    raise a
    # print(f'Final losses: no_torch: {loss_no_torch}, torch: {loss_torch}')
    
    return failed_once
            
def save_grad(name, grads):
    def hook(grad):
        grads[name] = grad
    return hook

# --------------------- Classes -------------------- 

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
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride)
    def forward(self, x):
        x = self.conv1(x)
        return x

class ConvolutionLinearTorchTestSmall(nn.Module):
    """Torch module for testing convolution and a linear layer"""
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride)
        self.fc1 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.hidden_dim)
        x = self.fc1(x)
        return x

class ConvolutionLinearTorchTestBig(nn.Module):
    """Torch module for testing two convolutions and two linear layers"""
    def __init__(self,in_channels, out_channels_1, 
                 out_channels_2, kernel_size_1, 
                 kernel_size_2, stride_1, stride_2,  
                 hidden_dim_1,hidden_dim_2, out_dim):
        super().__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, 
                               kernel_size=kernel_size_1, 
                               stride=stride_1)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, 
                               kernel_size=kernel_size_2, 
                               stride=stride_2)
        self.fc1 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc2 = nn.Linear(hidden_dim_2, out_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.hidden_dim_1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class GradNet(nn.Module):
    """Torch module for testing gradients of 
    convolution and transpose convolution"""
    def __init__(self,conv1, conv2, t_conv1, t_conv2):
        super().__init__()
        self.grads = {}
        self.conv1 = conv1
        self.conv2 = conv2
        self.t_conv1 = t_conv1
        self.t_conv2 = t_conv2

    def forward(self, x):
        x.register_hook(save_grad('conv1', self.grads))
        x_1 = self.conv1(x)
        x_1.register_hook(save_grad('sigmoid_1', self.grads))
        x_1_sigmoid = torch.sigmoid(x_1)
        x_1_sigmoid.register_hook(save_grad('conv2', self.grads))
        x_2 = self.conv2(x_1_sigmoid)
        x_2.register_hook(save_grad('relu_1', self.grads))
        x_2_relu = F.relu(x_2)
        x_2_relu.register_hook(save_grad('t_conv1', self.grads))        
        x_3 = self.t_conv1(x_2_relu)
        x_3.register_hook(save_grad('relu_2', self.grads))
        x_3_relu = F.relu(x_3)
        x_3_relu.register_hook(save_grad('t_conv2', self.grads))
        x_4 = self.t_conv2(x_3_relu)
        x_4.register_hook(save_grad('sigmoid_2', self.grads))
        x_4_sigmoid = torch.sigmoid(x_4)
        return x_4_sigmoid

class GradNetConvOnly(nn.Module):
    """Torch module for testing gradients of 
    convolution"""
    def __init__(self,conv1, conv2):
        super().__init__()
        self.grads = {}
        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, x):
        
        x.register_hook(save_grad('conv1', self.grads))
        
        x_1 = self.conv1(x)
        x_1.register_hook(save_grad('relu_1', self.grads))
        
        x_1_relu = F.relu(x_1)
        x_1_relu.register_hook(save_grad('conv2', self.grads))
        
        x_2 = self.conv2(x_1_relu)

        return x_2