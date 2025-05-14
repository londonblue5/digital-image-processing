import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough
from skimage.draw import circle_perimeter


def draw_circles_binary(image_shape, centers, radii):                   # returns a binary image with circles given centers and radii, used for Circular Hough
    image = np.zeros(image_shape, dtype=np.uint8)
    for (y, x), r in zip(centers, radii):                               # (y, x) because disk expects that order
        rr, cc = circle_perimeter(int(y), int(x), int(r), shape=image_shape)
        image[rr, cc] = 1
    return image


# ------------------------------- Load basketball image -------------------------------
img_filename = "C:/users/basketball_large.png"                          # !!! set the filepath to the image path
img = Image.open(fp=img_filename)                                       # get the image
bw_img = img.convert("L")                                               # keep only the Luminance component of the image
img_array = np.array(bw_img).astype(float) / 255.0                      # obtain the underlying np array and normalize values in [0, 1]



# ------------------------------------------------------------------------ Sobel ------------------------------------------------------------------------

sobel_arr = sobel_edge(img_array, thres=0.2)                            # set the THRESHOLD and apply the sobel filter                                                      -----
sobel_arr_255 = (sobel_arr * 255).astype(np.uint8)                      # Bring values back to [0, 255]
sobel_img = Image.fromarray(sobel_arr_255)
sobel_img.show()                                                        # Show the sobel filtered image

# graph of number of detected edges for different threshold values in sobel method
th = np.arange(0, 5.2, 0.2)                                             # threshold values
number_of_edges = np.zeros_like(th)
for idx, i in enumerate(th):
    sobel_arr = sobel_edge(img_array, i)
    number_of_edges[idx] = sobel_arr.sum()
plt.semilogy(th, number_of_edges, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Number of Edges')
plt.title('Sobel method')
plt.grid(True)
plt.show()



# ------------------------------------------------------------------------ LoG ------------------------------------------------------------------------

log_arr = log_edge(img_array, sigma=3)                                  # set the SIGMA and apply the laplacian of gaussian filter                                          -----
log_arr_255 = (log_arr * 255).astype(np.uint8)                          # Bring values back to [0, 255]
log_img = Image.fromarray(log_arr_255)                                  # Get the image object
log_img.show()                                                          # Show the image

# graph of number of detected edges for different sigma values in log method
s = np.arange(1, 17, 5)                                                 # sigma values
number_of_edges = np.zeros_like(s)
for idx, s_ in enumerate(s):
    print(f"iteration for sigma = {s_}")
    log_arr = log_edge(img_array, s_)
    number_of_edges[idx] = log_arr.sum()
plt.semilogy(s, number_of_edges, marker='o')
plt.xlabel('Sigma')
plt.ylabel('Number of Edges')
plt.title('LoG method')
plt.grid(True)
plt.yticks([1e4, 1e5, 1e6], ['1e4', '1e5', '1e6'])
plt.show()



# ------------------------------------------------------------------------ Circular Hough ------------------------------------------------------------------------

img_array = img_array[::10, ::10]                                       # undersampling 1/10, timeconsuming otherwise
dim = np.array([25, 25, 30])                                            # SET a, b, r, which are intervals x, y and radius are divided to search for circles                -----
R_max = min(img_array.shape) // 2                                       # max radius the algorithm will search for, not exceding the half of the smaller image dimension
V_min = 340                                                             # MINIMUM edge points for a specific center and radius to be considered circle                      -----

# Select one of the 2 following edge detector methods                                                                                                                       -----
edge_arr = sobel_edge(img_array, thres=0.3)                             # get input array with sobel filter
# edge_arr = log_edge(img_array, sigma=3)                               # get input array with log filter

print(f"Size of undersampled image: {img_array.shape}")
centers, radii = circ_hough(edge_arr, R_max, dim, V_min)                # apply the Hough filter to the edge filtered array of image
print(f"Found {len(radii)} circle{'s' if len(radii) != 1 else ''}")

circles_arr = draw_circles_binary(img_array.shape, centers, radii)      # call the function defined on top, get the binary image with circles as 1
img_array_255 = (img_array * 255).astype(np.uint8)                      # Scale original black and white image to [0, 255]
output_arr = np.stack([img_array_255]*3, axis=-1)                       # shape becomes (H, W, 3) for color
output_arr[circles_arr == 1] = [0, 255, 0]                              # Green circles are added where they were detected
output_img = Image.fromarray(output_arr)                                # Convert to PIL image and show
output_img.show()
