"""
References:
https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5
https://github.com/rmislam/PythonSIFT
http://weitz.de/sift/index.html?size=large
https://medium.com/@shartoo518/1-%E6%A6%82%E8%A7%88-5921fa471efb
https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
"""
import numpy as np
import cv2
from functools import cmp_to_key


# scale space and image pyramids
# base image
def generate_base_image(image, sigma, assumed_blur):
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)


# octave num
def compute_octave_num(image_shape):
    return int(round(np.log(min(image_shape)) / np.log(2) - 1))


# gaussian kernels
def generate_gaussian_kernels(sigma, num_intervals):
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1 / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma
    for index in range(1, num_images_per_octave):
        previous_sigma = (k ** (index - 1)) * sigma
        total_sigma = k * previous_sigma
        gaussian_kernels[index] = np.sqrt(total_sigma ** 2 - previous_sigma ** 2)
    return gaussian_kernels


# gaussian images
def generate_gaussian_images(image, num_octaves, gaussian_kernels):
    gaussian_images = []
    for index in range(num_octaves):
        octave_gaussian_images = [image]
        for kernel in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=kernel, sigmaY=kernel)
            octave_gaussian_images.append(image)
        gaussian_kernels.append(octave_gaussian_images)
        octave_base = octave_gaussian_images[-3]
        next_level_width = int(octave_base.shape[1] / 2)
        next_level_height = int(octave_base.shape[0] / 2)
        image = cv2.resize(octave_base, (next_level_width, next_level_height), interpolation=cv2.INTER_NEAREST)
    return np.array(gaussian_images)


# DoG images
def generate_DoG_images(gaussian_images):
    dog_images = []
    for octave_gaussian_images in gaussian_images:
        octave_dog_images = []
        for first_image, second_image in zip(octave_gaussian_images[:-1], octave_gaussian_images[1:]):
            octave_dog_images.append(np.subtract(second_image, first_image))
        dog_images.append(octave_dog_images)
    return np.array(dog_images)





