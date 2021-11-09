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


# check pixel extrema
def is_extrema(first_sub_image, second_sub_image, third_sub_image, threshold):
    center_pixel_value = second_sub_image[1, 1]
    if np.abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return np.all(center_pixel_value >= first_sub_image) and \
                   np.all(center_pixel_value >= third_sub_image) and \
                   np.all(center_pixel_value >= second_sub_image[0, :]) and \
                   np.all(center_pixel_value >= second_sub_image[2, :]) and \
                   center_pixel_value >= second_sub_image[1, 0] and \
                   center_pixel_value >= second_sub_image[1, 2]
        elif center_pixel_value < 0:
            return np.all(center_pixel_value <= first_sub_image) and \
                   np.all(center_pixel_value <= third_sub_image) and \
                   np.all(center_pixel_value <= second_sub_image[0, :]) and \
                   np.all(center_pixel_value <= second_sub_image[2, :]) and \
                   center_pixel_value <= second_sub_image[1, 0] and \
                   center_pixel_value <= second_sub_image[1, 2]
    return False


# compute center pixel gradient
def compute_gradient(pixel_array):
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return np.array([dx, dy, ds])


# compute center pixel hessian
def compute_hessian(pixel_array):
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] + pixel_array[1, 1, 0] - 2 * center_pixel_value
    dyy = pixel_array[1, 2, 1] + pixel_array[1, 0, 1] - 2 * center_pixel_value
    dss = pixel_array[2, 1, 1] + pixel_array[0, 1, 1] - 2 * center_pixel_value
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])


# localize extrema
def localize_extrema(i, j, image_index, octave_index, num_intervals,
                     octave_dog_images, sigma, contrast_threshold,
                     image_border_width, eigen_value_ratio=10, num_attempts=5):
    location_outside_image = False
    image_shape = octave_dog_images[0].shape
    for attempt_index in range(num_attempts):
        first_image, second_image, third_image = octave_dog_images[image_index-1:image_index+2]
        pixel_cube = np.stack([first_image[i-1: i+2, j-1: j+2],
                               second_image[i-1: i+2, j-1: j+2],
                               third_image[i-1: i+2, j-1: j+2]]).astype("float32") / 255
        gradient = compute_gradient(pixel_cube)
        hessian = compute_hessian(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.abs(extremum_update[0]) < 0.5 and np.abs(extremum_update[1]) < 0.5 and np.abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index = int(round(extremum_update[2]))
        if i < image_border_width or i >= image_shape[0] - image_border_width or \
            j < image_border_width or j >= image_shape[1] - image_border_width:
            location_outside_image = True
            break
    if location_outside_image:
        print("out side image")
        return None
    if attempt_index >= num_attempts - 1:
        print("Exceed num attempts")
        return None
    value = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
    if np.abs(value) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = np.trace(xy_hessian)
        xy_hessian_det =  np.linalg.det(xy_hessian)
        if xy_hessian_det > 0 and \
                eigen_value_ratio * (xy_hessian_trace ** 2) < ((eigen_value_ratio + 1) ** 2) * xy_hessian_trace:
            key_point = cv2.KeyPoint()
            key_point.pt = ((j + extremum_update[0]) * (2 ** octave_index),
                            (i + extremum_update[1]) * (2 ** octave_index))
            key_point.octave = octave_index + image_index * (2 ** 8) + \
                               int(round(extremum_update[2]+0.5) * 255) * 2 ** 16 #???
            key_point.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * \
                             (2 ** (octave_index + 1))
            key_point.response = np.abs(value)
            return key_point, image_index
    return None


# compute orientations
def compute_key_points_with_orientations():
    pass


# extrema
def find_scale_space_extrema(gaussian_images, dog_images, num_intervals,
                             sigma, image_border_width, contrast_threshold=0.04):
    threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # ???
    key_points = []
    for octave_index, octave_dog_images in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(octave_dog_images,
                                                                                   octave_dog_images[1:],
                                                                                   octave_dog_images[2:])):
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if is_extrema(first_image[i-1:i+2, j-1:j+2],
                                  second_image[i-1:i+2, j-1:j+2],
                                  third_image[i-1:i+2, j-1:j+2],
                                  threshold):
                        localization_result = localize_extrema(i, j, image_index+1, octave_index,
                                                           num_intervals, octave_dog_images,
                                                           sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            key_point, localized_image_index = localization_result
                            key_points_with_oreintations = compute_key_points_with_orientations(key_point,
                                                                                                octave_index,
                                                                                                gaussian_images[octave_index][localized_image_index])
                            for key_point in key_points_with_orientations:
                                key_points.append(key_point)
    return key_points









