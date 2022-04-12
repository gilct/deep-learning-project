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
        """Constructor  

        Parameters
        ----------
        params : ?
            Parameters
        """
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
        """Sequential constructor 
              
        Parameters
        ----------
        params : ?
            Parameters
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
        
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
        raise NotImplementedError 
        
    def param(self):
        """Returns the trainable parameters of the sequential model

        Returns
        -------
        list
            The list of trainable parameters
        """
        return []

# --------------------- Layers --------------------- 

class Conv2d(Module):
    """
    Convolutional layer class

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
        """Convolution constructor
        
        Parameters
        ----------
        params : ?
            Parameters
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
        
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
        raise NotImplementedError 
        
    def param(self):
        """Returns the trainable parameters of the convolutional layer

        Returns
        -------
        list
            The list of trainable parameters
        """
        return []    

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

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    """
    
    def __init__(self):
        """ReLU constructor 
              
        Parameters
        ----------
        params : ?
            Parameters
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
        
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
        raise NotImplementedError 

class Sigmoid(Module):
    """
    Sigmoid activation function class

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    """
    
    def __init__(self):
        """Sigmoid constructor 
              
        Parameters
        ----------
        params : ?
            Parameters
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
        
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
        raise NotImplementedError 

# ----------------------- Loss --------------------- 

class MSE(Module):
    """
    MSE loss class

    Methods
    -------
    forward(input)
        Runs the forward pass
    backward(gradwrtoutput)
        Runs the backward pass
    """
    
    def __init__(self):
        """MSE loss constructor
              
        Parameters
        ----------
        params : ?
            Parameters
        """
        raise NotImplementedError
    
    def forward (self, *input):
        """MSE loss forward pass

        Parameters
        ----------
        input : torch.tensor
            The input

        Returns
        -------
        torch.tensor
            The result of applying the MSE
        """
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        """MSE loss backward pass

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

# -------------------- Optimizer ------------------- 

class SGD():
    """
    Stochastic gradient descent (SGD) class

    Methods
    -------
    step()
        Perform a SGD step
    zero_grad()
        Reset the trainable parameter gradients to zero
    """
    
    def __init__(self):
        """SGD constructor
              
        Parameters
        ----------
        params : ?
            Parameters
        """
        raise NotImplementedError
    
    def step(self):
        """Perform a SGD step"""
        raise NotImplementedError
        
    def zero_grad(self):    
        """Reset the trainable parameter gradients to zero"""
        raise NotImplementedError