from torch import arange, empty
from math import ceil

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

def nn_upsampling(tensor, scale_factor=2):
    """Nearest neighbor upsampling

    Parameters
    ----------
    tensor : torch.tensor
        The tensor to perform nearest neighbor upsampling on
    scale_factor : int, optional
        Factor by which to scale the input's size (default is 2)

    Returns
    -------
    torch.tensor
        The tensor with the nearest neighbor upsampling applied to it
    """
    old_size = tensor.size()
    row_idx = ((arange(1, 1 + int(old_size[2]*scale_factor))/scale_factor).ceil() - 1).long()
    col_idx = ((arange(1, 1 + int(old_size[3]*scale_factor))/scale_factor).ceil() - 1).long()
    return tensor[:, :, :, col_idx][:, :, row_idx, :]

# might not need, used by NN upsampling
def compute_scaling_factor(output_dim, input_dim):
    output_h, output_w = output_dim
    input_h, input_w = input_dim
    return max(ceil(output_h / input_h), ceil(output_w / input_w))

# might not need, used by NN upsampling
def compute_upsampling_dim(output_dim, input_dim, scale_factor):
    output_h, output_w = output_dim
    input_h, input_w = input_dim
    ker_h = scale_factor * input_h - output_h + 1
    ker_w = scale_factor * input_w - output_w + 1 
    return (ker_h, ker_w)

# used by TransposeConv2d
def stride_tensor(tensor, stride=(1,1)):
    """fill zeros between tensor elements.
    e.g.: [[1 , 2]      [[1 ,0, 2]
           [3 , 4]] -->  [0 ,0, 0]
                         [3 ,0, 4]]
    """
    n,c,h,w = tensor.shape
    s_h, s_w = stride[0]-1, stride[1]-1
    ret = empty(n,c,h+(h-1)*s_h,w+(w-1)*s_w).fill_(0.0)
    ret[:,:,::(1+s_h),::(1+s_w)] = tensor
    return ret

def unstride_tensor(tensor, stride=(1,1)):
    """Revert the effect of stride_tensor"""
    return tensor[:,:,::stride[0],::stride[1]]

# used by TransposeConv2d
def pad_tensor_old(tensor, kernel_size, padding=(0,0)):
    n,c,h,w = tensor.shape
    k_h, k_w = kernel_size[0], kernel_size[1]
    padding_h, padding_w = padding
    new_h, new_w = h + 2 * (k_h - 1) - 2*padding_h, w + 2 * (k_w - 1) - 2*padding_w
    ret = empty(n,c,new_h,new_w).fill_(0.0)
    ret[:,:,k_h-1-padding_h:new_h-k_h+1+padding_h,k_w-1-padding_w:new_w-k_w+1+padding_w] = tensor
    return ret

def pad_tensor(t, kernel_size, padding=(0,0)):
    
    n,c,h,w = t.shape
    k_h, k_w = kernel_size[0], kernel_size[1]
    padding_h, padding_w = padding
    new_h, new_w = h + 2 * (k_h - 1) - 2*padding_h, w + 2 * (k_w - 1) - 2*padding_w
    ret = empty(n,c,new_h,new_w).fill_(0.0)

    start_h = max(k_h-1-padding_h,0)
    end_h = start_h + min(h, new_h)
    start_w = max(k_w-1-padding_w,0)
    end_w = start_w + min(w, new_w)

    t_h_offset = max(0,padding_h-k_h+1)
    t_w_offset = max(0,padding_w-k_w+1)
    ret[:,:,start_h:end_h,start_w:end_w] = t[:,:,t_h_offset:h-t_h_offset,t_w_offset:w-t_w_offset]
    return ret

def unpad_tensor(tensor, origi_size):
    ret = empty(origi_size).fill_(0.0)
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
    ret[:,:,start_h_origi:end_h_origi,start_w_origi:end_w_origi] = tensor[:,:,start_h_pad:end_h_pad,start_w_pad:end_w_pad]
    return ret