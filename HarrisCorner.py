"""
References:
https://github.com/TheAlgorithms/Python/blob/master/computer_vision/harris_corner.py

"""

import cv2
import numpy as np


class HarrisCorner(object):

    def __init__(self, k: float, kernel_size: int):
        if k in (0.04, 0.06):
            self.k = k
            self.kernel_size = kernel_size
        else:
            raise ValueError("invalid k value")

    def __str__(self):
        return f"Harris Corner Detection with k: {self.k}"

    def detect(self, img_path: str):
        img = cv2.imread(img_path, 0)
        height, width = img.shape
        corner_list = []
        color_img = img.copy()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)
        dy, dx = np.gradient(img)
        ixx = dx ** 2
        iyy = dy ** 2
        ixy = dx * dy
        k = self.k
        offset = self.kernel_size // 2
        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                wxx = ixx[y-offset: y+offset+1, x-offset: x+offset+1].sum()
                wyy = iyy[y-offset: y+offset+1, x-offset: x+offset+1].sum()
                wxy = ixy[y-offset: y+offset+1, x-offset: x+offset+1].sum()
                det = (wxx * wyy) - (wxy ** 2)
                trace = wxx + wyy
                r = det - k * (trace ** 2)
                if r > 0.5:
                    corner_list.append([x, y, r])
                    color_img.itemset((y, x, 0), 0)
                    color_img.itemset((y, x, 1), 0)
                    color_img.itemset((y, x, 2), 255)
        return color_img, corner_list


if __name__ == "__main__":
    test_path = "./test_img.jpg"
    corner_detect = HarrisCorner(0.04, 3)
    color_img = corner_detect.detect(test_path)
    cv2.imwrite("detect.jpg", corner_detect)