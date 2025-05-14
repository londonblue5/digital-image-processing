import numpy as np
from fir_conv import fir_conv

def sobel_edge(
        in_img_array: np.ndarray,
        thres: float
)-> np.ndarray:
    
    # Define Sobel masks
    Gx1 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=float)

    Gx2 = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=float)

    # Compute convolution using fir_conv
    grad_x = fir_conv(in_img_array, Gx1)
    grad_y = fir_conv(in_img_array, Gx2)

    # gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # threshhold for edges, 1 -> edge, 0 -> no edge
    out_img_array = (grad_mag >= thres).astype(int)

    return out_img_array
