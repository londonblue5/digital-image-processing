import numpy as np
from typing import Dict
from hist_utils import *

def perform_hist_modification(
        img_array: np.ndarray,
        hist_ref: Dict,
        mode: str
) -> np.ndarray:
    hist_of_img = calculate_hist_of_img(img_array, True)        # sorted dictionary of image histogram
    h_i = np.array([[k, v] for k, v in hist_of_img.items()])    # histogram of image as a 2d ndarray               
    h_r = np.array([[k, v] for k, v in hist_ref.items()])       # histogram of reference image as a 2d ndarray
    m_a = h_i.copy()                                            # modification array

    if mode == "greedy":                                        # greedy algorithm
        j = 0
        sum = 0
        for i in range(len(h_i)):                               # histogram modification loop
            sum = sum + h_i[i][1]
            m_a[i][1] = h_r[j][0]
            if sum >= h_r[j][1]:
                j = j + 1
                sum = 0
        
        m_dict = dict(m_a)
        modified_img = apply_hist_modification_transform(img_array, m_dict)
        return modified_img
    
    elif mode == "non-greedy":                                  # non-greedy algorithm
        j = 0
        sum = 0
        for i in range(len(h_i)):
            sum = sum + h_i[i][1]/2
            m_a[i][1] = h_r[j][0]
            if sum > h_r[j][1]:
                j = j + 1
                if j == h_r.shape[0]:
                    break
                sum = 0
            else:
                sum = sum + h_i[i][1]/2    

        m_dict = dict(m_a)
        modified_img = apply_hist_modification_transform(img_array, m_dict)
        return modified_img
    
    elif mode == "post-disturbance":                            # post-disturbance algorithm
        d = h_i[1][0] - h_i[0][0]
        v = np.random.uniform(-d/2, d/2, size = img_array.shape)
        i_hat = img_array + v                                               # add noise
        h_hat = calculate_hist_of_img(i_hat, True)
        return perform_hist_modification(i_hat, hist_ref, mode="greedy")    # return greedy modified 2d ndarray of image

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose between 'greedy', 'non-greedy', 'post-disturbance'.")

# histogram equalization
def perform_hist_eq(
        img_array: np.ndarray,
        mode: str,
        Lg: int
) -> np.ndarray:
    g_min = 0
    g_max = 1
    inv_Lg = 1/Lg
    g_dict = {}
    for i in range(Lg):
        g_value = g_min + i * (g_max - g_min) / (Lg - 1)
        g_dict[g_value] = inv_Lg
    return perform_hist_modification(img_array, g_dict, mode)

# histogram matching
def perform_hist_matching(
        img_array: np.ndarray,
        img_array_ref: np.ndarray,
        mode: str
) -> np.ndarray:
    h_ref = calculate_hist_of_img(img_array_ref, True)
    return perform_hist_modification(img_array, h_ref, mode)