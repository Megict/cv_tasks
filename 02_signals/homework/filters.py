import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    # print(image.shape)

    ### YOUR CODE HERE
    for cur_pix_x in range(0, Wi):
        for cur_pix_y in range(0, Hi):
            res_pix = 0
            for filt_pos_x in range(0, Wk):
                for filt_pos_y in range(0, Hk): 
                    img_pos_x = cur_pix_x + (int(0.5*Wk) - filt_pos_x)
                    img_pos_y = cur_pix_y + (int(0.5*Wk) - filt_pos_y)

                    img_pix_value = image[img_pos_y, img_pos_x] if img_pos_x >=0 and img_pos_y >= 0 and img_pos_x < Wi and img_pos_y < Hi else 0

                    res_pix += img_pix_value*kernel[filt_pos_y, filt_pos_x]
            out[cur_pix_y, cur_pix_x] = res_pix
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros_like(image)

    ### YOUR CODE HERE
    out = np.zeros((image.shape[0] + pad_height*2, image.shape[1] + pad_width*2))
    out[pad_height : pad_height + image.shape[0], pad_width : pad_width + image.shape[1]] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    kernel = np.flip(kernel,1)
    pad_img = zero_pad(image, int(Hk / 2), int(Wk / 2))
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.mean(np.multiply(pad_img[i:i+Hk][:,j:j+Wk], kernel))

    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    out = np.real(np.fft.ifft2(np.multiply(np.fft.fft2(image),np.fft.fft2(kernel, image.shape))))
    ### END YOUR CODE

    return out

from tqdm import tqdm

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    out = np.zeros_like(f, dtype = float)
    # ### YOUR CODE HERE
    g_mean = np.mean(g)
    # g = g - g_mean
    for m in tqdm(range(Hf - Hg)):
        for n in range(Wf - Wg):
            
            matr_frag = f[m : m + Hg, n : n + Wg]
            matr_mean = np.mean(matr_frag)
            matr_frag = matr_frag - matr_mean

            elm_matr = np.multiply(g, matr_frag)
            
            out[m,n] = np.sum(elm_matr) 

    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    out = np.zeros_like(f, dtype = float)
    # ### YOUR CODE HERE
    g_mean = np.mean(g)
    g = g - g_mean
    print(f"template mean value (after correction) : {np.mean(g)}")
    for m in tqdm(range(Hf - Hg)):
        for n in range(Wf - Wg):
            
            matr_frag = f[m : m + Hg, n : n + Wg]
            matr_mean = np.mean(matr_frag)
            matr_frag = matr_frag - matr_mean

            elm_matr = np.multiply(g, matr_frag)
            
            out[m,n] = np.sum(elm_matr)

    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    out = np.zeros_like(f, dtype = float)
    # ### YOUR CODE HERE
    g_mean = np.mean(g)
    g_std = np.std(g)
    g = g - g_mean
    print(f"template mean value (after correction) : {np.mean(g)}")
    for m in tqdm(range(Hf - Hg)):
        for n in range(Wf - Wg):
            
            matr_frag = f[m : m + Hg, n : n + Wg]
            matr_mean = np.mean(matr_frag)
            matr_std = np.std(matr_frag)
            matr_frag = matr_frag - matr_mean

            elm_matr = np.multiply(g, matr_frag)
            
            out[m,n] = np.sum(elm_matr) / (matr_std * g_std)

    ### END YOUR CODE

    return out
