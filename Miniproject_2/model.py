import math
from torch import empty
from torch.nn.functional import fold, unfold
from others.others import make_tuple, \
                          compute_conv_output_shape, \
                          compute_upsampling_dim, \
                          compute_scaling_factor, \
                          nn_upsampling, \
                          stride_tensor, \
                          pad_tensor, \
                          unpad_tensor, \
                          unstride_tensor
import torch
torch.set_grad_enabled(False)

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

# ------------------- Sequential -------------------

class Sequential(Module):
    """
    Sequential module class

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
        return self.params

# --------------------- Layers --------------------- 

class Linear(Module):
    """
    Linear module class

    Attributes
    ----------
    Weight : torch.tensor 
        the weight tensor of the linear module
    bias : torch.tensor
        the bias tensor of the linear module
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
        self.Weight = empty(input_size, output_size)
        self.bias = empty(output_size)
        self.grad_Weight = empty(input_size, output_size)
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
        return self.input.mm(self.Weight).add(self.bias)
        
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
        self.grad_Weight.add_(self.input.t().mm(grad))
        self.grad_bias.add_(grad.sum(0))
        return grad.mm(self.Weight.t()) 
        
    def param(self):
        """Returns the trainable parameters of the linear module

        Returns
        -------
        list
            The list of trainable parameters and their gradients
        """
        return [(self.Weight, self.grad_Weight), (self.bias, self.grad_bias)]

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
    Weight : torch.tensor 
        the weight tensor of the convolution module
    bias : torch.tensor
        the bias tensor of the convolution module
    grad_Weight: torch.tensor
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
        # transpose:
        #   [Out_channels, In_channels * H_Ker * W_Ker] 
        #       --> 
        #   [In_channels * H_Ker * W_Ker, Out_channels]
        Weight_reshaped = self.Weight.view(self.out_channels, -1).t()

        # matmul : 
        #   [In_channels * H_Ker * W_Ker, Out_channels]
        #       X 
        #   [Batch_size, Out_channels, H_out * W_out] 
        #       --> 
        #   [Batch_size, In_channels * H_Ker * W_Ker, H_out * W_out] 
        grad_wrt_input = Weight_reshaped \
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
        return [(self.Weight, self.grad_Weight), (self.bias, self.grad_bias)]

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
            self.Weight.fill_(init_val)
            self.bias.fill_(init_val)
        else:
            k = self.in_channels if not transpose else self.out_channels
            for k_size in self.kernel_size:
                k *= k_size
            stdv = 1. / math.sqrt(k)
            self.Weight.uniform_(-stdv, stdv)
            self.bias.uniform_(-stdv, stdv) 
        self.grad_Weight.zero_()
        self.grad_bias.zero_() 

# This module can be thrown away
# backprop not implemented
class NearestUpsampling(Module):
    """
    Upsampling layer class implemented using
    nearest neighbor upsampling and convolution.

    Attributes
    ----------
    input_dim : int
        dimensions (h,w) of the input image
    out_dim : int
        dimensions (h,w) of the upsampled image
    in_channels : int
        number of channels in the input image
    out_channels : int
        number of channels produced by the upsampling

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    param()
        Returns the list of trainable parameters
    """    
    def __init__(self, input_dim, out_dim, in_channels, 
                 out_channels, init_val=None):
        """Upsampler constructor
        
        Parameters
        ----------
        input_dim : int
            Dimensions (h,w) of the input image
        out_dim : int
            Dimensions (h,w) of the upsampled image
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the upsampling
        init_val: float, optional
            Default value of all weights (default is None)
        """
        self.scale_factor = compute_scaling_factor(out_dim, 
                                                   input_dim)
        ker_h, ker_w = compute_upsampling_dim(out_dim, 
                                              input_dim, 
                                              self.scale_factor)
        self.conv = Conv2d(in_channels,
                           out_channels,
                           (ker_h, ker_w),
                           init_val=init_val)

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
        return self.conv.forward(nn_upsampling(input[0].clone(), self.scale_factor))

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
        return self.conv.backward(gradwrtoutput) 
        
    def param(self):
        """Returns the trainable parameters of the upsampling layer

        Returns
        -------
        list
            The list of trainable parameters
        """
        return self.conv.param()

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
        zero padding to be added on both sides of input
    dilation : tuple
        controls the stride of elements within the neighborhood
    conv : Module
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
        return self.input.clamp(0)
        # return input[0].relu() # (using relu(), is this allowed?)
        
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
        backward = self.input.sign().clamp(0)
        grad = gradwrtoutput[0].clone().view(self.input.shape) # make sure grad is same shape as input
        # backward = self.input.relu().sign() # (using relu(), is this allowed?)
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
        return input[0].sigmoid() # Using sidmoid() here, is this allowed or do we need to implement from scratch?
        
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
        backward = self.input.sigmoid() * (1 - self.input.sigmoid()) # Using sidmoid() here, is this allowed or do we need to implement from scratch?
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
    
    def forward (self, prediction, target):
        """MSE loss module forward pass

        Parameters
        ----------
        prediction : torch.tensor
            The predicted values
        target : torch.tensor
            The ground truth values

        Returns
        -------
        torch.tensor
            The result of applying the MSE loss
        """      
        self.error = prediction - target
        return self.error.pow(2).mean()
        
    def backward(self):
        """MSE loss module backward pass

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
    nb_epochs : int
        the number of epochs to train for
    batch_size : int
        the size of the batches to use during training
    model : Module
        the network to use for training and predicting
    optimizer : SGD
        the optimizer to use during training
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

    def __init__(self) -> None:
        """Model constructor"""
        channels = 3
        out_ch_conv1, out_ch_conv2 = 32, 64
        kernel_size = (2,2)
        stride = 2
        lr, self.batch_size = 1e-5, 10
        self.model = Sequential(Conv2d(in_channels=channels, 
                                       out_channels=out_ch_conv1, 
                                       kernel_size=kernel_size, 
                                       stride=stride),
                                ReLU(),
                                Conv2d(in_channels=out_ch_conv1, 
                                       out_channels=out_ch_conv2, 
                                       kernel_size=kernel_size, 
                                       stride=stride),
                                ReLU(),
                                TransposeConv2d(in_channels=out_ch_conv2,
                                                out_channels=out_ch_conv1,
                                                kernel_size=kernel_size,
                                                stride=stride),
                                ReLU(),
                                TransposeConv2d(in_channels=out_ch_conv1,
                                                out_channels=channels,
                                                kernel_size=kernel_size,
                                                stride=stride),
                                Sigmoid())
        self.optimizer = SGD(self.model.param(), lr=lr)
        self.criterion = MSE()

    def load_pretrained_model(self) -> None:
        """Loads the parameters saved in bestmodel.pth into the model"""
        pass

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
        for e in range(num_epochs):
            item = f'\rTraining epoch {e+1}/{num_epochs}...'
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
        return self.model.forward(test_input)

def example_run():

    SEED = 2022
    torch.manual_seed(SEED)

    model = Model()
    num_epochs = 10

    # Use synthetic up data
    # Parameters of distribution of inputs and targets
    # in_channels, height, width = 3, 32, 32
    # n_samples = 200
    # mean, std = 0, 1
    # train_input = torch.randint(0,50,(n_samples, in_channels, height, width)).type(torch.FloatTensor)
    # train_targets = train_input + empty(n_samples, in_channels, height, width).normal_(mean, std)
    # some_sample = torch.randint(0,50,(1, in_channels, height, width)).type(torch.FloatTensor)

    # Use project data
    noisy_imgs_1, noisy_imgs_2 = torch.load("../data/train_data.pkl")
    noisy_imgs , clean_imgs = torch.load("../data/val_data.pkl")
    noisy_imgs_1, noisy_imgs_2 = torch.Tensor.float(noisy_imgs_1), torch.Tensor.float(noisy_imgs_2)
    noisy_imgs, clean_imgs = torch.Tensor.float(noisy_imgs), torch.Tensor.float(clean_imgs)
    train_input = noisy_imgs_1[:200]
    train_targets = noisy_imgs_2[:200]
    some_sample = noisy_imgs[0][None, :]

    # Normalize data
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)

    # Train and predict
    model.train(train_input, train_targets, num_epochs)

    # Evaluate
    some_sample.sub_(mu).div_(std)
    pred = model.predict(some_sample)
    # print(pred)

example_run()
