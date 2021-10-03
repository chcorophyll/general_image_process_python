"""
References:
https://github.com/TheAlgorithms/Python/blob/master/digital_image_processing/filters/convolve.py
https://github.com/TheAlgorithms/Python/blob/master/digital_image_processing/filters/gaussian_filter.py
"""

import numpy as np
import cv2


def img2col(image, block_size):
    rows, cols = image.shape
    dst_height = rows - block_size[0] + 1
    dst_width = cols - block_size[1] + 1
    image_array = np.zeros((dst_height * dst_width, block_size[0] * block_size[1]))
    row = 0
    for i in range(0, dst_height):
        for j in range(0, dst_width):
            window = np.ravel(image[i: i + block_size[0], j: j + block_size[1]])
            image_array[row, :] = window
            row += 1
    return image_array


def img_convolve(image, filter_kernel):
    height, width = image.shape[0], image.shape[1]
    k_size = filter_kernel.shape[0]
    pad_size = k_size // 2
    padding_img = np.pad(image, pad_size, mode="edge")
    image_array = img2col(padding_img, (k_size, k_size))
    kernel_array = np.ravel(filter_kernel)
    convolved_img = np.dot(image_array, kernel_array).reshape(height, width)
    return convolved_img


def generate_gaussian_kernel(k_size, sigma):
    center = k_size // 2
    x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
    g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    g = g / g.sum()
    return g


def gaussian_filter(image, k_size=3, sigma=1):
    kernel_gaussian = generate_gaussian_kernel(k_size, sigma)
    dst_gaussian = img_convolve(image, kernel_gaussian)
    # dst = dst_gaussian.astype(np.unit8)
    return dst_gaussian


if __name__ == "__main__":
    test_path = r"./test.jpg"
    img = cv2.imread(test_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_img = gaussian_filter(gray_img)
    gaussian_img = gaussian_img.astype(np.unit8)
    # show result images
    cv2.imshow("gaussian filter", gaussian_img)
    cv2.waitKey(0)