"""
References:
https://github.com/TheAlgorithms/Python/blob/master/digital_image_processing/filters/convolve.py

"""
import numpy as np
import cv2


def median_filter(image, kernel_size=3):
    offset = kernel_size // 2
    median_image = np.zeros_like(image)
    for i in range(offset, image.shape[0] - offset):
        for j in range(offset, image.shape[0] - offset):
            window = np.ravel(image[i-offset: i+offset+1, j-offset: j+offset+1])
            median = np.sort(window)[kernel_size * kernel_size // 2]
            median_image[i][j] = median[mask * mask // 2]
    return median_image


if __name__ == "__main__":
    test_path = r"./test.jpg"
    img = cv2.imread(test_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_img = median_filter(gray_img)
    # show result images
    cv2.imshow("median filter", median_img)
    cv2.waitKey(0)