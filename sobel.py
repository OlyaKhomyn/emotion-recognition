import numpy as np


mask = [[1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]]


def horizontal_sobel_filter(image, alpha=1, beta=0):
    new_img = np.zeros(image.shape, dtype=np.uint8)

    for row in range(1, image.shape[0]-1):
        for col in range(1, image.shape[1]-1):
            conv = image[row-1][col-1] * mask[0][0] + image[row-1][col] * mask[0][1] + image[row - 1][col + 1] * mask[0][2] + \
                   image[row][col-1] * mask[1][0] + image[row][col] * mask[1][1] + image[row][col+1] * mask[1][2] + \
                   image[row+1][col-1] * mask[2][0] + image[row+1][col] * mask[2][1] + image[row+1][col+1] * mask[2][2]

            new_img[row][col] = np.clip(abs(int(conv) * alpha + beta), 0, 255)

    new_img[new_img < 100] = 0

    return new_img
