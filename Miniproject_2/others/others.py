from torch import arange

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
    row_idx = ((arange(1, 1 + int(old_size[0]*scale_factor))/scale_factor).ceil() - 1).long()
    col_idx = ((arange(1, 1 + int(old_size[1]*scale_factor))/scale_factor).ceil() - 1).long()
    return tensor[:, col_idx][row_idx, :]