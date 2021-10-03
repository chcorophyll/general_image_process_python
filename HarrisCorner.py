"""
References:
https://github.com/TheAlgorithms/Python/blob/master/computer_vision/harris_corner.py
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/corner.py
"""

import cv2
import numpy as np
from .GaussianFilter import gaussian_filter


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


class HarrisCorner(object):

    def __init__(self, k: float, kernel_size: int):
        if k in (0.04, 0.06):
            self.k = k
            self.kernel_size = kernel_size
        else:
            raise ValueError("invalid k value")

    def __str__(self):
        return f"Harris Corner Detection with k: {self.k}"

    def sobel(self, img):
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        dst_x = np.abs(img_convolve(image, kernel_x))
        dst_y = np.abs(img_convolve(image, kernel_y))
        dst_x = dst_x * 255 / np.max(dst_x)
        dst_y = dst_y * 255 / np.max(dst_y)
        return dst_x, dst_y

    def gaussian(self, img, k_size=3, sigma=1):
        return gaussian_filter(img, k_size, sigma)

    def detect(self, img_path: str):
        img = cv2.imread(img_path, 0)
        height, width = img.shape
        corner_list = []
        color_img = img.copy()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)
        # origin method
        # dy, dx = np.gradient(img)
        # scikit-image method
        # sobel
        sobel_x, sobel_y = self.sobel(img)
        ixx = sobel_x ** 2
        iyy = sobel_y ** 2
        ixy = sobel_x * sobel_y
        # gaussian
        ixx = gaussian_filter(ixx)
        iyy = gaussian_filter(iyy)
        ixy = gaussian_filter(ixy)
        k = self.k
        offset = self.kernel_size //
        response = []
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                wxx = ixx[y-offset: y+offset+1, x-offset: x+offset+1].sum()
                wyy = iyy[y-offset: y+offset+1, x-offset: x+offset+1].sum()
                wxy = ixy[y-offset: y+offset+1, x-offset: x+offset+1].sum()
                # harris method
                det = (wxx * wyy) - (wxy ** 2)
                trace = wxx + wyy
                r = det - k * (trace ** 2)
                response.append(r)
                if r > 0.5:
                    corner_list.append([x, y, r])
                    color_img.itemset((y, x, 0), 0)
                    color_img.itemset((y, x, 1), 0)
                    color_img.itemset((y, x, 2), 255)
                # shi_tomasi
                # r = ((wxx + wyy) - np.sqrt((wxx - wyy) ** 2 + 4 * wxy ** 2)) / 2
                # if r > 0.5:
                #     corner_list.append([x, y, r])
                #     color_img.itemset((y, x, 0), 0)
                #     color_img.itemset((y, x, 1), 0)
                #     color_img.itemset((y, x, 2), 255)
        response = np.asarray(response).reshape((img.shape))

        return response, color_img, corner_list


if __name__ == "__main__":
    test_path = "./test_img.jpg"
    corner_detect = HarrisCorner(0.04, 3)
    color_img = corner_detect.detect(test_path)
    cv2.imwrite("detect.jpg", corner_detect)