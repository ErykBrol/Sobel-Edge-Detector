import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image


def sobel_edge_detect(input):
    # Init the x and y gradient kernels
    # These will be used for determining the "direction" of pixels
    # which is useful at borders between white/black pixels
    Gx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

    Gy = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])

    output_image = np.zeros((input.shape[0], input.shape[1]), dtype=float)

    padded_image = np.pad(input, 1, 'constant', constant_values=(0, 0))

    # To make convolution easier, use numpy stride tricks to subdivide the original matrix
    input_strided = as_strided(
        padded_image,
        shape=(
            padded_image.shape[0] - Gx.shape[0] + 1,
            padded_image.shape[1] - Gx.shape[1] + 1,
            Gx.shape[0],
            Gx.shape[1],
        ),
        strides=(
            padded_image.strides[0],
            padded_image.strides[1],
            padded_image.strides[0],
            padded_image.strides[1],
        ),
        writeable=False,
    )

    i = 0
    j = 0

    for row in input_strided:
        for tile in row:
            # Apply the derivative kernels
            multiplied_x = np.multiply(Gx, tile)
            multiplied_y = np.multiply(Gy, tile)

            sum_x_gradient = np.sum(multiplied_x)
            sum_y_gradient = np.sum(multiplied_y)

            # Sum their result
            output_image[j, i] = np.hypot(sum_x_gradient, sum_y_gradient)
            i += 1
        i = 0
        j += 1

    # Normalize the output image
    output_image *= 255.0 / np.max(output_image)

    return output_image
