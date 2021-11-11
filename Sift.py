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


FLOAT_TOLERANCE = 1e-7
# 1 scale space and image pyramids
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

# 2 find, rectify, filter extrema and add orientation
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
        image_index += int(round(extremum_update[2]))
        if i < image_border_width or i >= image_shape[0] - image_border_width or \
            j < image_border_width or j >= image_shape[1] - image_border_width or \
                image_index < 1 or image_index > num_intervals:
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
def compute_key_points_with_orientations(key_point, octave_index, gaussian_image, radius_factor=3,
                                         num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    key_points_with_orientations = []
    image_shape = gaussian_image.shape
    scale = scale_factor * key_point.size / np.float32(2 ** (octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)
    for i in range(-radius, radius + 1):
        region_y = int(round(key_point.pt[1] / np.float32(2 ** octave_index))) + i
        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(key_point.pt[0] / np.float32(2 ** octave_index))) + j
                if 0 < region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * num_bins / 360.0))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude
    for bin in range(num_bins):
        smooth_histogram[bin] = (6 * raw_histogram[bin] +
                                 4 * (raw_histogram[bin - 1] + raw_histogram[(n + 1) % num_bins]) +
                                 raw_histogram[bin - 2] +
                                 raw_histogram[(bin + 2) % num_bins]) / 16.0
    orientation_max = np.max(smooth_histogram)
    peak_condition = np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
                                    smooth_histogram > np.roll(smooth_histogram, -1))
    orientation_peaks = np.where(peak_condition)[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index +
                                       0.5 * (left_value - right_value) /
                                       (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360.0 - interpolated_peak_index * 360 / num_bins
            if np.abs(orientation - 360.0) < FLOAT_TOLERANCE:
                orientation = 0
            new_key_point = cv2.KeyPoint(*key_point.pt, key_point.size,
                                         orientation, key_point.response, key_point.octave)
            key_points_with_orientations.append(new_key_point)
    return key_points_with_orientations


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
                            key_points_with_orientations = compute_key_points_with_orientations(key_point,
                                                                                                octave_index,
                                                                                                gaussian_images[octave_index][localized_image_index])
                            for key_point in key_points_with_orientations:
                                key_points.append(key_point)
    return key_points


# compare key_points
def compare_key_points(key_point_1, key_point_2):
    if key_point_1.pt[0] != key_point_2.pt[0]:
        return key_point_1.pt[0] - key_point_2.pt[0]
    if key_point_1.pt[1] != key_point_2.pt[1]:
        return key_point_1.pt[1] - key_point_2.pt[1]
    if key_point_1.size != key_point_2.size:
        return key_point_2.size - key_point_1.size
    if key_point_1.angle != key_point_2.angle:
        return key_point_1.angle - key_point_2.angle
    if key_point_1.response != key_point_2.response:
        return key_point_2.response - key_point_1.response
    if key_point_1.octave != key_point_2.octave:
        return key_point_2.octave - key_point_1.octave
    return key_point_2.class_id - key_point_1.class_id


# remove duplicates
def remove_duplicate_key_points(key_points):
    key_points.sort(key=cmp_to_key(compare_key_points))
    unique_key_points = [key_points[0]]
    for next_key_point in key_points[1:]:
        last_unique_key_point = unique_key_points[-1]
        if last_unique_key_point.pt[0] != next_key_point.pt[0] or \
            last_unique_key_point.pt[1] != next_key_point.pt[1] or \
            last_unique_key_point.size != next_key_point.size or \
            last_unique_key_point.angle != next_key_point.angle:
            unique_key_points.append(next_key_point)
    return unique_key_points


# convert coordinate
def convert_to_input_image_size(key_points):
    converted_points = []
    for key_point in key_points:
        key_point.pt = tuple(0.5 * np.array(key_point.pt))
        key_point.size *= 0.5
        key_point.octave = (key_point.octave & ~255) | ((key_point.octave - 1) & 255)
        converted_points.append(key_point)
    return converted_points


# generate descriptor
# unpack octave
def unpack_octave(key_point):
    octave = key_point.octave & 255  # ???
    layer = (key_point.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale


# generate descriptors
def generate_descriptors(key_points, gaussian_images, window_width=4,
                         num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    descriptors = []
    for key_point in key_points:
        octave, layer, scale = unpack_octave(key_point)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * np.array(key_point.pt)).astype("int")
        bins_per_degree = num_bins / 360.0
        angle = 360.0 - key_point.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))
        hist_width = scale_multiplier * 0.5 * scale * key_point.size
        half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))
        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rotation = col * sin_angle + row * cos_angle
                col_rotation = col * cos_angle - row * sin_angle
                row_bin = (row_rotation / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rotation / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rotation / hist_width) ** 2 +
                                                              (col_rotation / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list,
                                                                col_bin_list,
                                                                magnitude_list,
                                                                orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin,
                                                                            orientation_bin]).astype("int")
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, \
                                                               col_bin - col_bin_floor, \
                                                               orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins
            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)
            histogram_tensor[row_bin_floor+1, col_bin_floor+1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= np.max(np.linalg.norm(descriptor_vector), FLOAT_TOLERANCE)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return descriptors







