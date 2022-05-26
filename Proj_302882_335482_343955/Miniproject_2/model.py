try:
    from others.others import make_tuple, \
                              compute_conv_output_shape, \
                              stride_tensor, \
                              pad_tensor, \
                              unpad_tensor, \
                              unstride_tensor
except ImportError:
    from .others.others import make_tuple, \
                               compute_conv_output_shape, \
                               stride_tensor, \
                               pad_tensor, \
                               unpad_tensor, \
                               unstride_tensor
    
import math
from torch import empty
from torch.nn.functional import fold, unfold
import torch
torch.set_grad_enabled(False)
import pickle
from pathlib import Path

# --------------------- Module ---------------------

class Module(object):
    """
    Base Module class

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    to(device)
        Moves the module to the device
    state_dict()
        Returns the weights and biases
    load_state_dict(state)
        Loads the weights and biases
    set_weight(weight)
        Set the weight
    set_bias(weight)
        Set the bias
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
        pass
        
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
        pass 
        
    def param(self):
        """Returns the trainable parameters of the model

        Returns
        -------
        list
            The list of trainable parameters
        """
        return []

    def to(self, device):
        """Moves the module to the device

        Parameters
        ----------
        device : torch.device
            The device to to move the module to
        """
        pass

    def state_dict(self):
        """Returns the weights and biases in a
        dictionary

        Returns
        -------
        dict
            The dictionary of weights and biases of the module
        """
        pass

    def load_state_dict(self, state_dict):
        """Loads the weights and biases from the `state_dict`
        dictionary

        Parameters
        ----------
        state_dict : dict
            The dictionary of weights and biases of the module
        """  
        pass

    def set_weight(self, weight):
        """Set the weight of the module
        
        Parameters
        ----------
        weight : torch.tensor
            The weight
        """
        pass

    def set_bias(self, bias):
        """Set the bias of the module
        
        Parameters
        ----------
        bias : torch.tensor
            The bias
        """
        pass       

# ------------------- Sequential -------------------

class Sequential(Module):
    """
    Sequential module class

    Attributes
    ----------
    modules : list
        a list of the sequential module's modules stored 
        in forward sequential order

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    to(device)
        Moves the module to the device
    state_dict()
        Returns the weights and biases of the sub-modules
    load_state_dict(state)
        Loads the weights and biases
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
    
    def forward(self, *input):
        """Sequential forward pass
        Calls forward(input) on all the modules 
        in sequential order

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
        Calls backward(grad) on all the modules 
        in reverse sequential order

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output
        """
        backward_data = gradwrtoutput[0].clone()
        for module in reversed(self.modules):
            backward_data = module.backward(backward_data) 
        
    def param(self):
        """Returns the trainable parameters of 
        the sequential module

        Returns
        -------
        list
            The list of trainable parameters
        """
        params = []
        for module in self.modules:
            for param in module.param():
                params.append(param)
        return params

    def to(self, device):
        """Moves the module to the device

        Parameters
        ----------
        device : torch.device
            The device to to move the module to
        """
        for module in self.modules:
            module.to(device)

    def state_dict(self):
        """Returns the weights and biases of the sub-modules
        in a dictionary

        Returns
        -------
        dict
            The dictionary of weights and biases of all the sub-modules
        """
        description = ["weight", "bias"]
        ret_dict = {}
        for i, module in enumerate(self.modules):
            for d_i, param in enumerate(module.param()):
                ret_dict[str(i)+"."+description[d_i]] = param[0]
        return ret_dict        

    def load_state_dict(self, state):
        """Loads the weights and biases from the `state`
        dictionary

        Parameters
        ----------
        state_dict : dict
            The dictionary of weights and biases of the sub-modules
        """ 
        for i, module in enumerate(self.modules):
            w_key = str(i)+"."+"weight"
            b_key = str(i)+"."+"bias"
            if w_key in state:
                module.set_weight(state[w_key])
            if b_key in state:
                module.set_bias(state[b_key])

# --------------------- Layers --------------------- 

class Linear(Module):
    """
    Linear module class

    Attributes
    ----------
    weight : torch.tensor 
        the weight tensor of the linear module
    bias : torch.tensor
        the bias tensor of the linear module
    grad_weight: torch.tensor
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
        Resets the trainable parameters of 
        the linear module
    """

    def __init__(self, input_size, output_size, init_val=None):
        """Linear module constructor
        
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
        self.weight = empty(input_size, output_size)
        self.bias = empty(output_size)
        self.grad_weight = empty(input_size, output_size)
        self.grad_bias = empty(output_size)
        self.reset_params(input_size, init_val)
    
    def forward(self, *input):
        """Linear module forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying the linear module
        """
        self.input = input[0].clone().flatten(start_dim=1) # Need to flatten (e.g. if input comes from conv layer)
        return self.input.mm(self.weight).add(self.bias)
        
    def backward(self, *gradwrtoutput):
        """Linear module backward pass

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
        self.grad_weight.add_(self.input.t().mm(grad))
        self.grad_bias.add_(grad.sum(0))
        return grad.mm(self.weight.t()) 
        
    def param(self):
        """Returns the trainable parameters of the linear module

        Returns
        -------
        list
            The list of trainable parameters and their gradients
        """
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]

    def reset_params(self, input_size, init_val=None):
        """Resets the trainable parameters of the linear module
        
        Parameters
        ----------
        input_size : int
            The size of the input
        init_val: float, optional
            Default value of all weights (default is None)
        """
        if init_val is not None:
            self.weight.fill_(init_val)
            self.bias.fill_(init_val)
        else:
            stdv = 1. / math.sqrt(input_size)
            self.weight = self.weight.uniform_(-stdv, stdv)
            self.bias = self.bias.uniform_(-stdv, stdv) 
        self.grad_weight.zero_()
        self.grad_bias.zero_()

class Conv2d(Module):
    """
    Convolution module class

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
    weight : torch.tensor 
        the weight tensor of the convolution module
    bias : torch.tensor
        the bias tensor of the convolution module
    grad_weight: torch.tensor
        the gradient of the loss wrt the weight matrix 
    grad_bias: torch.tensor
        the gradient of the loss wrt the bias 
    in_shape : torch.size
        the shape of the input to the module
    out_shape : torch.size
        the shape of the output of the module
    input = torch.tensor
        the input to the module

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    reset_params(transpose, init_val)
        Resets the trainable parameters of the 
        convolution module
    to(device)
        Moves the module to the device
    set_weight(weight)
        Set the weight
    set_bias(weight)
        Set the bias
    """   

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, transpose=False,
                 init_val=None):
        """Convolution module constructor
        
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
        transpose: boolean, optional
            Determines if the convolution is transpose 
            or not (default is False)
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

        self.weight = empty(out_channels, 
                            in_channels, 
                            self.kernel_size[0], 
                            self.kernel_size[1])
        self.bias = empty(out_channels)
        self.grad_weight = empty(out_channels, 
                                 in_channels, 
                                 self.kernel_size[0], 
                                 self.kernel_size[1])
        self.grad_bias = empty(out_channels)
        self.reset_params(transpose, init_val)

        self.in_shape = None
        self.out_shape = None
        self.input = None
    
    def forward(self, *input):
        """Convolution module forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying the convolution
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
        convolved = self.weight.view(self.out_channels, -1) \
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
        """Convolutional module backward pass

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
        grad_weight = grad_reshaped.mm(self.input)

        # reshape : 
        #   [Out_channels, In_channels * H_Ker * W_Ker] 
        #       --> 
        #   [Out_channels, In_channels, H_Ker, W_Ker]
        self.grad_weight.add_(grad_weight.reshape(self.grad_weight.shape))

        # ---------------- Gradient wrt input -----------------

        # view : 
        #   [Out_channels, In_channels, H_Ker, W_Ker] 
        #       --> 
        #   [Out_channels, In_channels * H_Ker * W_Ker]
        # transpose:
        #   [Out_channels, In_channels * H_Ker * W_Ker] 
        #       --> 
        #   [In_channels * H_Ker * W_Ker, Out_channels]
        weight_reshaped = self.weight.view(self.out_channels, -1).t()

        # matmul : 
        #   [In_channels * H_Ker * W_Ker, Out_channels]
        #       X 
        #   [Batch_size, Out_channels, H_out * W_out] 
        #       --> 
        #   [Batch_size, In_channels * H_Ker * W_Ker, H_out * W_out] 
        grad_wrt_input = weight_reshaped \
                         .matmul(grad.flatten(start_dim=2))

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

        # For test purposes only
        self.grad = grad_wrt_input

        return grad_wrt_input
        
    def param(self):
        """Returns the trainable parameters of the 
        convolutional module

        Returns
        -------
        list
            The list of trainable parameters and their gradients
        """
        return [(self.weight, self.grad_weight), (self.bias, self.grad_bias)]

    def reset_params(self, transpose=False, init_val=None):
        """Resets the trainable parameters of the convolutional layer
        according to the documentation of nn.Con2d and nn.ConvTranspose2d
        
        Parameters
        ----------
        transpose: boolean, optional
            Determines if the convolution is transpose 
            or not (default is False)
        init_val: float, optional
            Default value of all weights (default is None)
        """
        if init_val is not None:
            self.weight.fill_(init_val)
            self.bias.fill_(init_val)
        else:
            k = self.in_channels if not transpose else self.out_channels
            for k_size in self.kernel_size:
                k *= k_size
            stdv = 1. / math.sqrt(k)
            self.weight.uniform_(-stdv, stdv)
            self.bias.uniform_(-stdv, stdv) 
        self.grad_weight.zero_()
        self.grad_bias.zero_() 

    def to(self, device):
        """Moves the module to the device

        Parameters
        ----------
        device : torch.device
            The device to to move the module to
        """
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)
        self.grad_weight = self.grad_weight.to(device)
        self.grad_bias = self.grad_bias.to(device)

    def set_weight(self, weight):
        """Set the weight of the convolution module
        
        Parameters
        ----------
        weight : torch.tensor
            The weight
        """
        self.weight = weight

    def set_bias(self, bias):
        """Set the weight of the convolution module
        
        Parameters
        ----------
        bias : torch.tensor
            The bias
        """
        self.bias = bias

class TransposeConv2d(Module): 
    """
    Transpose convolution module class implemented using
    border zero padding and convolution.

    Attributes
    ----------
    kernel_size : tuple
        size of the convolving kernel
    stride : tuple
        stride of the convolution
    padding : tuple
        implicit zero padding to be added on both sides of input
    dilation : tuple
        controls the stride of elements within the neighborhood
    conv : Conv2d
        the convolution module
    strided_shape : torch.size
        the shape of the input after it has been strided

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    to(device)
        Moves the module to the device
    set_weight(weight)
        Set the weight
    set_bias(weight)
        Set the bias
    """    
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=0,
                 dilation=1, init_val=None):
        """Transpose convolution constructor
        
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the transpose
            convolution
        kernel_size : int or tuple
            Size of the convolving kernel
        stride : int or tuple, optional
            Stride of the transpose convolution (default is 1)
        padding : int or tuple, optional
            Implicit zero padding to be added on both sides of 
            input (default is 0)
        dilation : int or tuple, optional
            Controls the stride of elements within the 
            neighborhood (default is 1)
        init_val: float, optional
            Default value of all weights (default is None)
        """   
        super().__init__()         
        self.kernel_size = make_tuple(kernel_size)
        self.stride = make_tuple(stride)
        self.padding = make_tuple(padding)
        self.dilation = make_tuple(dilation)
        self.conv = Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=self.kernel_size,
                           dilation=self.dilation,
                           transpose=True,
                           init_val=init_val)
        self.strided_shape = None

    def forward(self, *input):
        """Transpose convolution module forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying transpose convolution
        """
        strided = stride_tensor(input[0].clone(), self.stride)
        self.strided_shape = strided.shape
        pad_k_h = self.kernel_size[0] + (self.kernel_size[0]-1)*(self.dilation[0]-1)
        pad_k_w = self.kernel_size[1] + (self.kernel_size[1]-1)*(self.dilation[1]-1)
        padded = pad_tensor(strided, (pad_k_h, pad_k_w), self.padding)
        return self.conv.forward(padded)

    def backward(self, *gradwrtoutput):
        """Transpose convolution module backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        grad = gradwrtoutput[0].clone()
        grad_from_conv = self.conv.backward(grad)
        unpadded_grad = unpad_tensor(grad_from_conv,
                                     self.strided_shape)
        unstrided_grad = unstride_tensor(unpadded_grad, 
                                         self.stride)

        # For test purposes only
        self.grad = unstrided_grad

        return unstrided_grad
        
    def param(self):
        """Returns the trainable parameters of the transpose
        comvolution module

        Returns
        -------
        list
            The list of trainable parameters and their gradients
        """
        return self.conv.param()

    def to(self, device):
        """Moves the module to the device

        Parameters
        ----------
        device : torch.device
            The device to to move the module to
        """
        self.conv.to(device)

    def set_weight(self, weight):
        """Set the weight of the transpose convolution module
        
        Parameters
        ----------
        weight : torch.tensor
            The weight
        """
        self.conv.set_weight(weight)

    def set_bias(self, bias):
        """Set the bias of the transpose convolution module
        
        Parameters
        ----------
        bias : torch.tensor
            The bias
        """
        self.conv.set_bias(bias)

class Upsampling(TransposeConv2d):
    """
    Wrapper for transpose convolution module class
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=0,
                 dilation=1, init_val=None):
        """Upsampling constructor
        
        Parameters
        ----------
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the transpose
            convolution
        kernel_size : int or tuple
            Size of the convolving kernel
        stride : int or tuple, optional
            Stride of the transpose convolution (default is 1)
        padding : int or tuple, optional
            Implicit zero padding to be added on both sides of 
            input (default is 0)
        dilation : int or tuple, optional
            Controls the stride of elements within the 
            neighborhood (default is 1)
        init_val: float, optional
            Default value of all weights (default is None)
        """  
        super().__init__(in_channels,out_channels,
                         kernel_size,stride,
                         padding,dilation,init_val)

# --------------- Activation functions ------------- 

class ReLU(Module):
    """
    ReLU activation function module

    Attributes
    ----------
    input : torch.tensor 
        the input to the ReLU module

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    """
    
    def __init__(self):
        """ReLU module constructor """
        super().__init__()
        self.input = None
    
    def forward(self, *input):
        """ReLU module forward pass

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
        return input[0].relu()
        
    def backward(self, *gradwrtoutput):
        """ReLU module backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        grad = gradwrtoutput[0].clone().view(self.input.shape) # make sure grad is same shape as input
        backward = self.input.relu().sign()
        ret = backward.mul(grad)

        # For test purposes only
        self.grad = ret

        return ret

class Sigmoid(Module):
    """
    Sigmoid activation function module

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
        """Sigmoid module constructor"""
        super().__init__()
        self.input = None
    
    def forward(self, *input):
        """Sigmoid module forward pass

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
        return input[0].sigmoid()
        
    def backward(self, *gradwrtoutput):
        """Sigmoid module backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output

        Returns
        -------
        torch.tensor
            The gradient of the loss wrt the module's input
        """
        backward = self.input.sigmoid() * (1 - self.input.sigmoid())
        grad = gradwrtoutput[0].clone().view(self.input.shape) # make sure grad is same shape as input
        ret = backward.mul(grad)

        # For test purposes only
        self.grad = ret

        return ret

# ----------------------- Loss --------------------- 

class MSE(Module):
    """
    MSE loss module

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
        """MSE loss module constructor"""
        super().__init__()
        self.error = None
    
    def forward (self, *input):
        """MSE loss module forward pass

        Parameters
        ----------
        input : torch.tensor
            The predicted and target values

        Returns
        -------
        torch.tensor
            The result of applying the MSE loss
        """   
        prediction, target = input[0], input[1]   
        self.error = prediction - target
        return self.error.pow(2).mean()
        
    def backward(self, *gradwrtoutput):
        """MSE loss module backward pass

        Parameters
        ----------
        gradwrtoutput : torch.tensor
            The gradients wrt the module's output
            Never used sice MSE is assumed to be last part
            of network

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

# ---------------------- Model --------------------- 

class Model():
    """
    Model class

    Attributes
    ----------
    batch_size : int
        the size of the batches to use during training
    model : Module
        the network to use for training and predicting
    optimizer : SGD
        the optimizer to use during training
    lr : float
        the learning rate to use during training
    criterion : Module
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

        SEED = 3
        torch.manual_seed(SEED)
        
        in_channels = 3
        conv_1_in, conv_1_out = in_channels, 32
        conv_2_in, conv_2_out = conv_1_out, 64
        t_conv_1_in, t_conv_1_out = conv_2_out, conv_2_in
        t_conv_2_in, t_conv_2_out = t_conv_1_out, in_channels

        kernel_size, stride, padding = (2,2), (1,1), (1,1)
        self.batch_size = 10
        self.lr = 10-4   

        self.model = Sequential(Conv2d(conv_1_in, 
                                       conv_1_out, 
                                       kernel_size=kernel_size, 
                                       stride=stride,
                                       padding=padding),
                                ReLU(),
                                Conv2d(conv_2_in, 
                                       conv_2_out, 
                                       kernel_size=kernel_size, 
                                       stride=stride,
                                       padding=padding),
                                ReLU(),
                                Upsampling(t_conv_1_in, 
                                           t_conv_1_out, 
                                           kernel_size=kernel_size, 
                                           stride=stride,
                                           padding=padding),
                                ReLU(),
                                Upsampling(t_conv_2_in, 
                                           t_conv_2_out, 
                                           kernel_size=kernel_size, 
                                           stride=stride,
                                           padding=padding),
                                Sigmoid())
        self.model.to(self.device)
        self.optimizer = SGD(self.model.param(), lr=self.lr)
        self.criterion = MSE()

    def load_pretrained_model(self) -> None:
        """Loads the parameters saved in bestmodel.pth into the model"""
        PATH_TO_MODEL = Path(__file__).parent / "bestmodel.pth"
        with open(PATH_TO_MODEL,'rb') as dict:
            # print("Loading with pickle!")
            best_model_state_dict = pickle.load(dict)
        # don't use torch !!
        # best_model_state_dict = torch.load(PATH_TO_MODEL)
        self.model.load_state_dict(best_model_state_dict)
        self.model.to(self.device)
        self.optimizer = SGD(self.model.param(), lr=self.lr)

    def train(self, train_input, train_target, num_epochs) -> None:
        """Runs the training procedure

        Parameters
        ----------
        train_input: (N, C, H, W), torch.tensor
            Tensor containing a noisy version of the images
        train_target: (N, C, H, W), torch.tensor
            Tensor containing another noisy version of the same 
            images, which only differs from the input by their noise
        num_epochs: int
            The number of epochs to train for
        """
        train_input = (train_input  / 255.0).float().to(self.device)
        train_target = (train_target  / 255.0).float().to(self.device)
        for e in range(num_epochs):
            item = f'\rTraining (on {self.device}) epoch {e+1}/{num_epochs}...'
            print(item, sep=' ', end='', flush=True)
            for input, targets in zip(train_input.split(self.batch_size),
                                      train_target.split(self.batch_size)):
                output = self.model.forward(input)
                loss = self.criterion.forward(output, targets)
                self.optimizer.zero_grad()
                self.model.backward(self.criterion.backward())
                self.optimizer.step()
        print()

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
        test_input = (test_input / 255.0).float().to(self.device)
        return self.model.forward(test_input) * 255.0
