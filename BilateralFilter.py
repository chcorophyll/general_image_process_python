"""
References:
https://github.com/TheAlgorithms/Python/blob/master/digital_image_processing/filters/bilateral_filter.py
"""
import cv2
import numpy as np


def vec_gaussian(img, sigma):
    cons = 1 / (sigma * np.sqrt(2 * np.pi))
    return cons * np.exp(-((img / sigma) ** 2) * 0.5)


def get_window(img, center, k_size=3):
    y, x = center
    off_set = k_size // 2
    return img[y-off_set: y+off_set+1, x-off_set:x+off_set+1]


def get_space_gaussian(k_size=3, sigma=1):
    space_gaussian = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            space_gaussian[i, j] = np.sqrt(abs(i - k_size // 2) ** 2 + abs(j - k_size // 2) ** 2)
    return vec_gaussian(space_gaussian, sigma)


def bilateral_filter(img, space_sigma=1, intensity_sigma=1, k_size=3):
    height, width = img.shape[0], img.shape[1]
    pad_size = k_size // 2
    padding_img = np.pad(img, pad_size, mode="edge")
    space_gaussian = get_space_gaussian(k_size, space_sigma)
    bilateral_img = np.zeros((height, width))
    for i in range(pad_size, padding_img.shape[0]-pad_size):
        for j in range(pad_size, padding_img.shape[1]-pad_size):
            window_intensity = get_window(padding_img, (i, j), k_size)
            differ_intensity = window_intensity - padding_img[i][j]
            intensity_gaussian = vec_gaussian(differ_intensity, intensity_sigma)
            current_gaussian = np.multiply(intensity_gaussian, space_gaussian)
            values = np.multiply(window_intensity, current_gaussian)
            current_value = np.sum(values) / np.sum(current_gaussian)
            bilateral_filter[i - pad_size][j - pad_size] = current_value
    return bilateral_img


if __name__ == "__main__":
    test_path = r"./test.jpg"
    img = cv2.imread(test_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img / 255
    gray_img = gray_img.astype("float32")
    bilateral_img = bilateral_filter(gray_img)
    bilateral_img = bilateral_img * 255
    bilateral_img = bilateral_img.astype("uint8")
    # show result images
    cv2.imshow("bilateral filter", bilateral_img)
    cv2.waitKey(0)

