import numpy as np
from typing import Dict


def calculate_hist_of_img(
        img_array: np.ndarray,                                              # dtype = float    grayscale
        return_normalized: bool
) -> Dict[float, float]:
    values, counts = np.unique(img_array, return_counts=True)               # get values and counts of image
    if return_normalized:
        counts = counts / counts.sum()
    frequency_dict: Dict[float, float] = {                                  # get relative frequencies dictionary
        key: value
        for key, value in zip(values, counts)
    }
    sorted_dict = dict(sorted(frequency_dict.items()))                      # sort the dictionary
    return sorted_dict


def apply_hist_modification_transform(
        img_array: np.ndarray,
        modification_transform: Dict
) -> np.ndarray:
    vectorized_map = np.vectorize(lambda x: modification_transform[x])      # transform function
    modif_array = vectorized_map(img_array)                                 # apply the transform
    return modif_array                                                      # returns modified image as 2d ndarray




"""
diction = calculate_hist_of_img([0.1, 0.1, 0.2, 0.3, 0.3, 0.3], True)
print(diction)
"""