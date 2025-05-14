import numpy as np
from scipy.signal import convolve2d

def fir_conv(
    in_img_array: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    
    # Flip kernel for convolution
    h_flipped = np.flip(h)
    
    # Convolve, output image same dimentions with input
    out_img_array = convolve2d(in_img_array, h_flipped, mode='same', boundary='fill', fillvalue=0)
    
    return out_img_array
