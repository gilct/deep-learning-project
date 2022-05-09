from torch import empty

def make_tuple(value, repetitions=2):
    """Helper function for generating a value as a
    tuple if it isn't already

    Parameters
    ----------
    value : int or tuple
        Value to create tuple from
    repetitions : int, optional
        Number of repetitons of `value` to include in tuple

    Returns
    -------
    tuple
        The generated tuple
    """
    return value if type(value) is tuple else tuple(value for _ in range(repetitions))

def compute_conv_output_shape(in_dim,
                              kernel_size,
                              stride,
                              padding,
                              dilation):
    """Computes the output shapes (height and width) of
    a convolution operation

    Parameters
    ----------
    in_dim : tuple
        Dimensions (h, w) of the input to the convolution
    kernel_size : tuple
        Size of the convolving kernel
    stride : tuple
        Stride of the convolution
    padding : tuple
        Zero padding to be added on both sides of input
    dilation : tuple
        Controls the stride of elements within the neighborhood

    Returns
    -------
    tuple
        The height and width of the result of the convolution
    """
    h_in, w_in = in_dim
    h_filter, w_filter = kernel_size
    h_stride, w_stride = stride
    h_padding, w_padding = padding
    h_dilation, w_dilation = dilation

    h_out = (h_in + 2 * h_padding - h_dilation * (h_filter - 1 ) - 1) // h_stride + 1
    w_out = (w_in + 2 * w_padding - w_dilation * (w_filter - 1 ) - 1) // w_stride + 1

    return (h_out, w_out)

# used by TransposeConv2d
def stride_tensor(tensor, stride=(1,1)):
    """Stride the input tensor.
    In this context, striding means inserting columns 
    and rows of zeros between the input columns and 
    rows. By default, `stride` is set to (1,1) meaning that 
    no insertions are made. See example below:
                                 [[1,0,0,2,0,0,3],
    [[1,2,3],                     [0,0,0,0,0,0,0],
     [4,5,6], -- stride(2,3) -->  [4,0,0,5,0,0,6],
     [7,8,9]]                     [0,0,0,0,0,0,0],
                                  [7,0,0,8,0,0,9]] 
    In the above example, we can see that with a stride
    of (2,3) an additional row of zeros has been inserted between
    the rows of the input tensor and an additional two columns 
    of zeros have been inserted between the columns of the input
    tensor.

    The purpose of this function is to modify the input tensor
    passed to a TransposeConv2d module so that it can then be passed
    to an ordinary Conv2d module uder the hood.
    
    Parameters
    ----------
    tensor : torch.tensor
        The tensor to stride
    stride : tuple, optional
        The stride to use for the height and width (default is (1,1))
    
    Returns
    -------
    torch.tensor
        The strided tensor
    """
    n,c,h,w = tensor.shape
    s_h, s_w = stride[0]-1, stride[1]-1
    ret = empty(n,c,h+(h-1)*s_h,w+(w-1)*s_w, device=tensor.device).fill_(0.0)
    ret[:,:,::(1+s_h),::(1+s_w)] = tensor
    return ret

def unstride_tensor(tensor, stride=(1,1)):
    """Revert the effect of stride_tensor.
    See stride_tensor().

    The purpose of this function is to modify the tensor of gradients 
    wrt the TransposeConv2d module's output. Since we use a Conv2d under 
    the hood of a TransposeConv2d module, the gradient returned from the
    Conv2d needs to be unstrided (and unpadded, see unpad_tensor()) before
    being relayed backward to the previous module.
    
    Parameters
    ----------
    tensor : torch.tensor
        The tensor to un-stride
    stride : tuple, optional
        The stride to use for the height and width (default is (1,1))
    
    Returns
    -------
    torch.tensor
        The un-strided tensor
    """
    return tensor[:,:,::stride[0],::stride[1]]

def pad_tensor(tensor, kernel_size, padding=(0,0)):
    """Zero pad the input tensor.
    The amounnt of zero padding to apply to the height and
    width of the input tensor is determined by the kernel
    size (`kernel_size`) and the implicit zero padding (`padding`).
    With no implicit zero padding, which is the default 
    behavior, the basic idea is to apply enough zero 
    padding such that the kernel's lower right cell is 
    applied to the upper left cell of the input tensor. 
    See example below:
                               [[0, 0, 0, 0, 0, 0, 0],
    [[1,2,3],                   [0, 0, 1, 2, 3, 0, 0],
     [4,5,6],  -- pad(2,3) -->  [0, 0, 4, 5, 6, 0, 0],
     [7,8,9]]                   [0, 0, 7, 8, 9, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]]
    In the above example, we can see that when the convolution
    starts, the kernel (which is 2 by 3) will start in the upper
    left corner and it's lower right cell will be placed above the
    input tensor's upper left cell (which contains the value 1).

    When `padding` is also used, the idea is to consider 
    the columns and rows at the edge of the input
    tensor as zero padding, meaning we discard them.
    See example below:
                               [[0, 0, 0],
    [[1,2,3],     pad(2,3)      [1, 2, 3],
     [4,5,6],    -- with -->    [4, 5, 6],
     [7,8,9]]   padding=(0,2)   [7, 8, 9],
                                [0, 0, 0]]
    In the above example, we can see that the additional implicit
    zero padding (0,2) removes two columns on the right and left 
    of the padded tensor (see first example).
    
    The purpose of this function is to modify the input tensor
    passed to a TransposeConv2d module so that it can then be passed
    to an ordinary Conv2d module uder the hood.
    
    Parameters
    ----------
    tensor : torch.tensor
        The tensor to pad
    kernel_size : tuple
        Size of the convolving kernel
    padding : tuple, optional
        The amount of implicit zero padding to use
        for the height and width (default is (0,0))
    
    Returns
    -------
    torch.tensor
        The padded tensor
    """
    n,c,h,w = tensor.shape
    k_h, k_w = kernel_size[0], kernel_size[1]
    padding_h, padding_w = padding
    new_h, new_w = h + 2 * (k_h - 1) - 2*padding_h, \
                   w + 2 * (k_w - 1) - 2*padding_w
    ret = empty(n,c,new_h,new_w, device=tensor.device).fill_(0.0)

    start_h = max(k_h-1-padding_h,0)
    end_h = start_h + min(h, new_h)
    start_w = max(k_w-1-padding_w,0)
    end_w = start_w + min(w, new_w)

    t_h_offset = max(0,padding_h-k_h+1)
    t_w_offset = max(0,padding_w-k_w+1)
    ret[:,:,start_h:end_h,start_w:end_w] = \
        tensor[:,:,t_h_offset:h-t_h_offset,t_w_offset:w-t_w_offset]
    return ret

def unpad_tensor(tensor, origi_size):
    """Revert the effect of zero padding applied to
    the input tensor. See pad_tensor().

    The purpose of this function is to modify the tensor of gradients 
    wrt the TransposeConv2d module's output. Since we use a Conv2d 
    under the hood of a TransposeConv2d module, the gradient returned 
    from the Conv2d needs to be unpadded (and unstrided, see unstride_tensor()) 
    before being relayed backward to the previous module.
    
    Parameters
    ----------
    tensor : torch.tensor
        The tensor to un-pad
    origi_size : tuple
        Shape to un-pad to, often the shape of the (strided) 
        tensor that was fed to pad_tensor() in TransposeConv2d
    
    Returns
    -------
    torch.tensor
        The un-padded tensor
    """
    ret = empty(origi_size, device=tensor.device).fill_(0.0)
    n,c,h_origi,w_origi = origi_size
    n,c,h_pad,w_pad = tensor.size()
    start_h_pad = max(int((h_pad - h_origi) / 2), 0)
    end_h_pad = min(h_origi + start_h_pad,h_pad)
    start_w_pad = max(int((w_pad - w_origi) / 2), 0)
    end_w_pad = min(w_origi + start_w_pad,w_pad)
    start_h_origi = max(int((h_origi - h_pad) / 2), 0)
    end_h_origi = min(h_pad + start_h_origi,h_origi)
    start_w_origi = max(int((w_origi - w_pad) / 2), 0)
    end_w_origi = min(w_pad + start_w_origi,w_origi)
    ret[:,:,start_h_origi:end_h_origi,start_w_origi:end_w_origi] = \
        tensor[:,:,start_h_pad:end_h_pad,start_w_pad:end_w_pad]
    return ret