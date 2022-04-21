import math
from torch import empty
from torch.nn.functional import fold, unfold
from others.others import make_tuple, compute_conv_output_shape

import torch
torch.set_grad_enabled(False)

# --------------------- Module ---------------------

class Module(object):
    """
    Module class defining model architecture

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    """
    
    def __init__(self):
        """Constructor"""
        pass
    
    def forward(self, *input):
        """Runs the forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying the module
        """
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        """Runs the backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        raise NotImplementedError 
        
    def param(self):
        """Returns the trainable parameters of the model

        Returns
        -------
        list
            The list of trainable parameters
        """
        return []

# ------------------- Sequential -------------------

class Sequential(Module):
    """
    Sequential model architecture class

    Attributes
    ----------
    modules : list
        a list of the sequential module's modules stored 
        in forward sequential order
    params : list
        a list of all the trainable parameters of the 
        sequential module

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    """
    
    def __init__(self, *modules):
        """Sequential constructor 
              
        Parameters
        ----------
        modules : Module
            Variable length list of modules
        """
        super().__init__()
        self.modules = list(modules)
        self.params = []
        for module in self.modules:
            for param in module.param():
                self.params.append(param)
    
    def forward(self, *input):
        """Sequential forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying the sequential model
        """
        forward_data = input[0].clone()
        for module in self.modules:
            forward_data = module.forward(forward_data)
        return forward_data
        
        
    def backward(self, *gradwrtoutput):
        """Sequential backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        backward_data = gradwrtoutput[0].clone()
        for module in reversed(self.modules):
            backward_data = module.backward(backward_data) 
        
    def param(self):
        """Returns the trainable parameters of the sequential model

        Returns
        -------
        list
            The list of trainable parameters
        """
        return self.params

# --------------------- Layers --------------------- 

class Linear(Module):
    """
    Linear layer class

    Attributes
    ----------
    Weight : torch.tensor 
        the weight tensor of the linear layer
    bias : torch.tensor
        the bias tensor of the linear layer
    grad_Weight: torch.tensor
        the gradient of the loss wrt the weight matrix 
    grad_bias: torch.tensor
        the gradient of the loss wrt the bias 

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    reset_params(input_size, init_val)
        Resets the trainable parameters of the linear layer
    """
    
    def __init__(self, input_size, output_size, init_val=None):
        """Convolution constructor
        
        Parameters
        ----------
        input_size : int
            Size of each input sample
        output_size : int
            Size of each input sample
        init_val: float, optional
            Default value of all weights (default is None)
        """
        super().__init__()
        self.Weight = empty(input_size, output_size)
        self.bias = empty(output_size)
        self.grad_Weight = empty(input_size, output_size)
        self.grad_bias = empty(output_size)
        self.reset_params(input_size, init_val)
    
    def forward(self, *input):
        """Linear layer forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying convolution
        """
        self.input = input[0].clone().flatten(start_dim=1) # Need to flatten (e.g. if input comes from conv layer)
        return self.input.mm(self.Weight).add(self.bias)
        
    def backward(self, *gradwrtoutput):
        """Linear layer backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the input
        """
        grad = gradwrtoutput[0].clone()
        self.grad_Weight.add_(self.input.t().mm(grad))
        self.grad_bias.add_(grad.sum(0))
        return grad.mm(self.Weight.t()) 
        
    def param(self):
        """Returns the trainable parameters of the linear layer

        Returns
        -------
        list
            The list of trainable parameters and their gradients
        """
        return [(self.Weight, self.grad_Weight), (self.bias, self.grad_bias)]

    def reset_params(self, input_size, init_val=None):
        """Resets the trainable parameters of the linear layer
        
        Parameters
        ----------
        input_size : int
            The size of the input
        init_val: float, optional
            Default value of all weights (default is None)
        """
        if init_val is not None:
            self.Weight.fill_(init_val)
            self.bias.fill_(init_val)
        else:
            stdv = 1. / math.sqrt(input_size)
            self.Weight = self.Weight.uniform_(-stdv, stdv)
            self.bias = self.bias.uniform_(-stdv, stdv) 
        self.grad_Weight.zero_()
        self.grad_bias.zero_()

class Conv2d(Module):
    """
    Convolutional layer class

    Attributes
    ----------
    in_channels : int
        number of channels in the input image
    out_channels : int
        number of channels produced by the convolution
    kernel_size : tuple
        size of the convolving kernel
    stride : tuple
        stride of the convolution
    padding : tuple
        zero padding to be added on both sides of input
    dilation : tuple
        controls the stride of elements within the neighborhood
    Weight : torch.tensor 
        the weight tensor of the convolutional layer
    bias : torch.tensor
        the bias tensor of the convolutional layer
    grad_Weight: torch.tensor
        the gradient of the loss wrt the weight matrix 
    grad_bias: torch.tensor
        the gradient of the loss wrt the bias 
    in_shape : torch.size
        the shape of the input to the module
    out_shape : torch.size
        the shape of the output of the module
    input = torch.tensor
        the input of the module

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    reset_params()
        Resets the trainable parameters of the 
        convolutional layer
    """    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, init_val=None):
        """Convolution constructor
        
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the convolution
        kernel_size : int or tuple
            Size of the convolving kernel
        stride : int or tuple, optional
            Stride of the convolution (default is 1)
        padding : int or tuple, optional
            Zero padding to be added on both sides of 
            input (default is 0)
        dilation : int or tuple, optional
            Controls the stride of elements within the 
            neighborhood (default is 1)
        init_val: float, optional
            Default value of all weights (default is None)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = make_tuple(kernel_size)
        self.stride = make_tuple(stride)
        self.padding = make_tuple(padding)
        self.dilation = make_tuple(dilation)

        self.Weight = empty(out_channels, 
                            in_channels, 
                            self.kernel_size[0], 
                            self.kernel_size[1])
        self.bias = empty(out_channels)
        self.grad_Weight = empty(out_channels, 
                                 in_channels, 
                                 self.kernel_size[0], 
                                 self.kernel_size[1])
        self.grad_bias = empty(out_channels)
        self.reset_params(init_val)

        self.in_shape = None
        self.out_shape = None
        self.input = None
    
    def forward(self, *input):
        """Convolutional layer forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying convolution
        """
        self.in_shape = input[0].shape
        batch_size, _ , h_in, w_in = self.in_shape

        # unfold : 
        #   [Batch_size, In_channels, H_in, W_in]
        #       -->
        #   [Batch_size, In_channels * H_Ker * W_Ker, H_out * W_out]
        unfolded = unfold(input[0].clone(), 
                          kernel_size=self.kernel_size, 
                          dilation=self.dilation, 
                          padding=self.padding, 
                          stride=self.stride)
        
        # transpose(1, 2) : 
        #   [Batch_size, In_channels * H_Ker * W_Ker, H_out * W_out] 
        #       --> 
        #   [Batch_size, H_out * W_out, In_channels * H_Ker * W_Ker]
        # flatten(end_dim=1) : 
        #   [Batch_size, H_out * W_out, In_channels * H_Ker * W_Ker]
        #       -->
        #   [Batch_size * H_out * W_out, In_channels * H_Ker * W_Ker]
        self.input = unfolded.clone().transpose(1, 2).flatten(end_dim=1)

        # view: 
        #   [Out_channels, In_channels, H_Ker, W_Ker] 
        #       --> 
        #   [Out_channels, In_channels * H_Ker * W_Ker]
        # matmul : 
        #   [Out_channels, In_channels * H_Ker * W_Ker] 
        #       X 
        #   [Batch_size, In_channels * H_Ker * W_Ker, H_out * W_out]
        #       --> 
        #   [Batch_size, Out_channels, H_out * W_out]
        convolved = self.Weight.view(self.out_channels, -1) \
                    .matmul(unfolded) \
                    .add(self.bias.view(1, -1, 1))

        h_out, w_out = compute_conv_output_shape((h_in, w_in),
                                                 self.kernel_size,
                                                 self.stride,
                                                 self.padding,
                                                 self.dilation)
        
        # view : 
        #   [Batch_size, Out_channels, H_out * W_out] 
        #       --> 
        #   [Batch_size, Out_channels, H_out, W_out]
        convolved = convolved.view(batch_size, 
                                   self.out_channels, 
                                   h_out, 
                                   w_out)
        self.out_shape = convolved.shape
                
        return convolved
        
    def backward(self, *gradwrtoutput):
        """Convolutional layer backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the input
        """

        grad = gradwrtoutput[0].clone()
        grad = grad.view(self.out_shape) # make sure grad has same shape as output
        
        # --------------- Gradient of bias term ---------------

        # Sum over all except out_channel dim
        self.grad_bias.add_(grad.sum(axis=(0, 2, 3)))

        # -------------- Gradient of weight term --------------  
  
        # transpose(0, 1) : 
        #   [Batch_size, Out_channels, H_out, W_out] 
        #       --> 
        #   [Out_channels, Batch_size, H_out, W_out]
        # reshape: 
        #   [Out_channels, Batch_size, H_out, W_out] 
        #       --> 
        #   [Out_channels, Batch_size * H_out * W_out]
        grad_reshaped = grad.transpose(0, 1).reshape(self.out_channels, -1)

        # mm : 
        #   [Out_channels, Batch_size * H_out * W_out] 
        #       X 
        #   [Batch_size * H_out * W_out, In_channels * H_Ker * W_Ker]
        #       -->
        #   [Out_channels, In_channels * H_Ker * W_Ker]
        grad_Weight = grad_reshaped.mm(self.input)

        # reshape : 
        #   [Out_channels, In_channels * H_Ker * W_Ker] 
        #       --> 
        #   [Out_channels, In_channels, H_Ker, W_Ker]
        self.grad_Weight.add_(grad_Weight.reshape(self.grad_Weight.shape))

        # ---------------- Gradient wrt input -----------------

        # view : 
        #   [Out_channels, In_channels, H_Ker, W_Ker] 
        #       --> 
        #   [Out_channels, In_channels * H_Ker * W_Ker]
        Weight_reshaped = self.Weight.view(self.out_channels, -1)

        # mm : 
        #   [In_channels * H_Ker * W_Ker, Out_channels] 
        #       X 
        #   [Out_channels, Batch_size * H_out * W_out]
        #       -->
        #   [In_channels * H_Ker * W_Ker, Batch_size * H_out * W_out]
        grad_wrt_input = Weight_reshaped.t().mm(grad_reshaped)

        # view : 
        #   [In_channels * H_Ker * W_Ker, Batch_size * H_out * W_out] 
        #       --> 
        #   [Batch_size, In_channels * H_Ker * W_Ker, H_out * W_out]
        grad_wrt_input = grad_wrt_input \
                         .view(self.in_shape[0], grad_wrt_input.shape[0],-1)

        # fold : 
        #   [Batch_size, In_channels * H_Ker * W_Ker, H_out * W_out] 
        #       --> 
        #   [Batch_size, In_channels, H_in, W_in]
        grad_wrt_input = fold(grad_wrt_input,
                              output_size=self.in_shape[2:],
                              kernel_size=self.kernel_size,
                              dilation=self.dilation,
                              padding=self.padding,
                              stride=self.stride)
        return grad_wrt_input
        
    def param(self):
        """Returns the trainable parameters of the convolutional layer

        Returns
        -------
        list
            The list of trainable parameters and their gradients
        """
        return [(self.Weight, self.grad_Weight), (self.bias, self.grad_bias)]

    def reset_params(self, init_val=None):
        """Resets the trainable parameters of the convolutional layer
        
        Parameters
        ----------
        init_val: 
            Default value of all weights (default is None)
        """
        if init_val is not None:
            self.Weight.fill_(init_val)
            self.bias.fill_(init_val)
        else:
            in_c = self.in_channels
            for k_size in self.kernel_size:
                in_c *= k_size
            stdv = 1. / math.sqrt(in_c)
            self.Weight.uniform_(-stdv, stdv)
            self.bias.uniform_(-stdv, stdv) 
        self.grad_Weight.zero_()
        self.grad_bias.zero_() 

class NearestUpsampling(Module):
    """
    Upsampling layer class

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    """    
    def __init__(self):
        """Upsampler constructor
        
        Parameters
        ----------
        params : ?
            Parameters
        """
        raise NotImplementedError
    
    def forward(self, *input):
        """Upsampling layer forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying upsampling
        """
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        """Upsampling layer backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        raise NotImplementedError 
        
    def param(self):
        """Returns the trainable parameters of the upsampling layer

        Returns
        -------
        list
            The list of trainable parameters
        """
        return []  

# --------------- Activation functions ------------- 

class ReLU(Module):
    """
    ReLU activation function class

    Attributes
    ----------
    input : torch.tensor 
        the input of the ReLU module

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    """
    
    def __init__(self):
        """ReLU constructor """
        super().__init__()
        self.input = None
    
    def forward(self, *input):
        """ReLU forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying the ReLU function
        """
        self.input = input[0].clone()
        return self.input.clamp(0)
        # return input[0].relu() # (using relu(), is this allowed?)
        
    def backward(self, *gradwrtoutput):
        """ReLU backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        backward = self.input.sign().clamp(0)
        grad = gradwrtoutput[0].clone().view(self.input.shape) # make sure grad is same shape as input
        # backward = self.input.relu().sign() # (using relu(), is this allowed?)
        return backward.mul(grad)

class Sigmoid(Module):
    """
    Sigmoid activation function class

    Attributes
    ----------
    input : torch.tensor 
        the input of the sigmoid module

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    """
    
    def __init__(self):
        """Sigmoid constructor"""
        super().__init__()
        self.input = None
    
    def forward(self, *input):
        """Sigmoid forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying the sigmoid function
        """
        self.input = input[0].clone()
        return input[0].sigmoid() # Using sidmoid() here, is this allowed or do we need to implement from scratch?
        
    def backward(self, *gradwrtoutput):
        """Sigmoid backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        backward = self.input.sigmoid() * (1 - self.input.sigmoid()) # Using sidmoid() here, is this allowed or do we need to implement from scratch?
        grad = gradwrtoutput[0].clone().view(self.input.shape) # make sure grad is same shape as input
        return backward.mul(grad)

# ----------------------- Loss --------------------- 

class MSE(Module):
    """
    MSE loss class

    Attributes
    ----------
    error : torch.tensor 
        the error between the predicted and target values

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    """
    
    def __init__(self):
        """MSE loss constructor"""
        super().__init__()
        self.error = None
    
    def forward (self, prediction, target):
        """MSE loss forward pass

        Parameters
        ----------
        prediction : torch.tensor
            The predicted values
        target : torch.tensor
            The ground truth values

        Returns
        -------
        torch.tensor
            The result of applying the MSE
        """      
        self.error = prediction - target
        return self.error.pow(2).mean()
        
    def backward(self):
        """MSE loss backward pass

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        return (2.0 / self.error.numel()) * self.error 

# -------------------- Optimizer ------------------- 

class SGD():
    """
    Stochastic gradient descent (SGD) class

    Attributes
    ----------
    params : iterable
        the list of trainable parameters to apply SGD to
    lr: float
        the learning rate
    
    Methods
    -------
    step()
        Perform a SGD step
    zero_grad()
        Reset the trainable parameter gradients to zero
    """
    
    def __init__(self, params, lr):
        """SGD constructor
              
        Parameters
        ----------
        params : iterable
            Iterable of parameters to optimize
        lr: float
            Learning rate
        """
        self.params = params
        self.lr = lr
    
    def step(self):
        """Perform a SGD step"""
        for param, grad in self.params:
            if (param is not None) and (grad is not None):
                param.sub_(grad, alpha=self.lr)
        
    def zero_grad(self):    
        """Reset the trainable parameter gradients to zero"""
        for param, grad in self.params:
            if (param is not None) and (grad is not None):
                grad.zero_()