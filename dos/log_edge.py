import numpy as np
from scipy.ndimage import gaussian_laplace

def log_edge(in_img_array: np.ndarray,
             sigma: np.float32
) -> np.ndarray:

    # Step 1: Laplacian of Gaussian filtering
    log_image = gaussian_laplace(in_img_array, sigma)

    # Step 2: Εντοπισμός zero-crossings
    out_img_array = np.zeros_like(in_img_array, dtype=int)

    for i in range(1, log_image.shape[0] - 1):
        for j in range(1, log_image.shape[1] - 1):
            region = log_image[i-1:i+2, j-1:j+2]
            center = log_image[i, j]

            # sign change -> Aij = 1
            if np.any(region * center < 0):
                out_img_array[i, j] = 1

    return out_img_array
