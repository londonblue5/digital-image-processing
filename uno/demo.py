from PIL import Image
import numpy as np
from hist_modif import perform_hist_eq, perform_hist_matching


img_filename = "C:/Users/gyat.jpg"                         # set the filepath to the image file
img = Image.open(fp=img_filename)                               # get the image
bw_img = img.convert("L")                                       # keep only the Luminance component of the image
img_array = np.array(bw_img).astype(float) / 255.0              # obtain the underlying np array

ref_img_filename = "C:/Users/ref_img.jpg"                       # set the filepath to the reference file
ref_img = Image.open(fp=ref_img_filename)                       # get the image
bw_ref_img = ref_img.convert("L")                               # keep only the Luminance component of the image
ref_img_array = np.array(bw_ref_img).astype(float) / 255.0      # obtain the underlying np array


# histogram equalization

eq_arr = perform_hist_eq(img_array, "post-disturbance", 6)      # Histogram equalization, arguements: image array, method, Lg (number of desired output levels)
eq_arr_255 = (eq_arr * 255).astype(np.uint8)                    # Bring values back to [0, 255]
mod_img = Image.fromarray(eq_arr_255)                           # Get the image object
mod_img.show()                                                  # Show the image


# histogram matching - uncomment to execute

"""
match_arr = perform_hist_matching(img_array, ref_img_array, "non-greedy")   # Histogram matching, arguements: image array, reference image array, method
match_arr_255 = (match_arr * 255).astype(np.uint8)                          # Bring values back to [0, 255]
mod_img = Image.fromarray(match_arr_255)                                    # Get the image object
mod_img.show()
"""