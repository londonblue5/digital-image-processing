import numpy as np

def circ_hough(
        in_img_array: np.ndarray,           # input image
        R_max: float,                       # max radius
        dim: np.ndarray,                    # step of x, y, r
        V_min: int                          # min value required to be considered circle, threshhold
) -> tuple[np.ndarray, np.ndarray]:         # centers and radii
    
    print(f"Total edge points: {np.sum(in_img_array)}")

    height, width = in_img_array.shape
    x_steps, y_steps, r_steps = dim

    x_step = int(width / x_steps)
    y_step = int(height / y_steps)
    r_step_size = R_max / r_steps           # radius step size as float

    centers = np.empty((0, 2), dtype = int)
    radii = np.empty((0,), dtype = float)

    pos_edge_points = np.argwhere(in_img_array == 1)
    
    # ----- Circular Hough -----
    for x_center in np.arange(0, width, x_step):
        print(f"Searching for possible x of center: {x_center}")
        for y_center in np.arange(0, height, y_step):
            count = np.zeros(r_steps, dtype=int)
            
            for y, x in pos_edge_points:
                dist = np.sqrt((x_center-x)**2 + (y_center-y)**2)
                if dist > R_max + r_step_size:
                    continue
                i_radius = int(dist/r_step_size)
                if 0 <= i_radius < r_steps:
                    count[i_radius] += 1
            
            for i in range(r_steps):
                if count[i] >= V_min:    
                    centers = np.vstack([centers, [y_center, x_center]])
                    radii = np.append(radii, r_step_size*i + r_step_size/2)

    return centers, radii