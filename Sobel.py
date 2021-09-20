"""
References:
https://github.com/TheAlgorithms/Python/blob/master/digital_image_processing/filters/convolve.py

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


def sobel_filter(image):
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    dst_x = np.abs(img_convolve(image, kernel_x))
    dst_y = np.abs(img_convolve(image, kernel_y))
    dst_x = dst_x * 255 / np.max(dst_x)
    dst_y = dst_y * 255 / np.max(dst_y)
    dst_xy = np.sqrt((np.square(dst_x) + np.square(dst_y)))
    dst_xy = dst_xy * 255 / np.max(dst_xy)
    dst = dst_xy.astype(np.unit8)
    theta = np.arctan2(dst_y, dst_x)
    return dst, theta


if __name__ == "__main__":
    test_path = r"./test.jpg"
    img = cv2.imread(test_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_gradient, sobel_theta = sobel_filter(gray_img)
    # show result images
    cv2.imshow("sobel filter", sobel_gradient)
    cv2.imshow("sobel theta", sobel_theta)
    cv2.waitKey(0)