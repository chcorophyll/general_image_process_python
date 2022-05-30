"""
References:
https://github.com/TheAlgorithms/Python/blob/master/computer_vision/harris_corner.py
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/corner.py
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/orb.py
"""
import numpy as np
import cv2
from .GaussianFilter import gaussian_filter
from .FastCorner import detect, calculate_score
from .HarrisCorner import HarrisCorner


class ORB(object):

    def __init__(self, down_scale=1.2, n_scales=8,
                 n_key_points=500, fast_n=9, fast_threshold=0.08, harris_k=0.04):
        self.down_scale = down_scale
        self.n_scales = n_scales
        self.n_key_points = n_key_points
        self.fast_n = fast_n
        self.fast_threshold = fast_threshold
        self.harris_k = harris_k
        self.key_points = None
        self.scales = None
        self.responses = None
        self.orientations = None
        self.descriptors = None
        self.load_positions()

    def load_positions(self):
        self.positions = np.loadtxt("orb_descriptor_positions.txt", dtype=np.int8)
        self.positions_0 = self.positions[:, :2]
        self.positions_1 = self.positions[:, 2:]

    def bresenham_circle_mask(self):
        mask = np.zeros((31, 31))
        mask_umax = [15, 15, 15, 15, 15, 14, 14, 14, 13, 13, 13, 11, 10, 9, 8, 6, 3]
        for i in range(-15, 16):
            for j in range(-mask_umax[abs(i)], mask_umax[abs(i)]+1):
                mask[15+j, 15+i] = 1
        return mask

    def pyramid_reduce(self, image, down_scale=2, sigma=None):
        out_shape = tuple([d // float(down_scale) for d in image.shape][::-1])
        if sigma is None:
            sigma = 2 * down_scale / 6.0
        gaussian_image = gaussian_filter(image, k_size=int(6*sigma+0.5), sigma=sigma)  # 99% k = 2 * (3 * sigma) + 1
        resized_image = cv2.resize(gaussian_image, out_shape)
        return resized_image

    def pyramid_gaussian(self, image, max_layer=-1, down_scale=2, sigma=None):
        layer = 0
        current_shape = image.shape
        previous_layer_image = image
        yield image
        while layer != max_layer:
            layer += 1
            layer_image = self.pyramid_reduce(previous_layer_image, down_scale, sigma)
            previous_shape = current_shape
            previous_layer_image = layer_image
            current_shape = layer_image.shape
            if np.all(current_shape == previous_shape):
                break
            yield layer_image

    def mask_border_key_points(self, image_shape, key_points, distance=16):
        rows = image_shape[0]
        cols = image_shape[1]
        mask = (((distance - 1) < key_points[:, 0]) &
                ((rows - distance + 1) > key_points[:, 0]) &
                ((distance - 1) < key_points[:, 1]) &
                ((cols - distance + 1) > key_points[:, 1]))
        return mask

    def fast_corner(self, image, threshold, fast_n):
        return detect(image, threshold=threshold, fast_n=fast_n)

    def corner_orientations(self, image, key_points, bresenham_mask):
        mask_rows = bresenham_mask.shape[0]
        mask_cols = bresenham_mask.shape[1]
        mask_rows_1 = (mask_rows - 1) // 2
        mask_cols_1 = (mask_cols - 1) // 2
        padded_image = np.pad(image, (mask_rows_1, mask_cols_1), mode="constant", constant_values=0)
        orientations = np.zeros(key_points.shape[0])
        for i in range(key_points.shape[0]):
            raw_row = key_points[i, 0]
            raw_col = key_points[i, 1]
            m_01 = 0
            m_10 = 0
            for row in range(mask_rows):
                temp_m_01 = 0
                for col in range(mask_cols):
                    if bresenham_mask[row][col]:
                        current = padded_image[raw_row+row, raw_col+col]
                        m_10 += current * (col - mask_cols_1)
                        temp_m_01 += current
                m_01 += temp_m_01 * (row - mask_rows_1)
            orientations[i] = np.arctan2(m_01, m_10)
        return orientations

    def harris_corner(self, image):
        detector = HarrisCorner(self.harris_k, kernel_size=5)
        return detector.detect(image)

    def detect_octave(self, image):
        key_points, _ = self.fast_corner(image, self.fast_threshold, self.fast_n)
        if len(key_points) == 0:
            return np.zeros((0, 2)), np.zeros((0, )), np.zeros((0, ))
        # fast_response = []
        # for key_point, region_of_interest in zip(key_points, regions_of_interest):
        #     response = calculate_score(image, key_point, region_of_interest)
        #     fast_response.append(response)
        key_points = np.asarray(key_points)
        # fast_response = np.asarray(fast_response)
        mask = self.mask_border_key_points(image.shape, key_points)
        key_points = key_points[mask]
        bresenham_mask = self.bresenham_circle_mask()
        orientations = self.corner_orientations(image, key_points, bresenham_mask)
        responses, _, _ = self.harris_corner(image)
        responses = responses[key_points[:, 0], key_points[:, 1]]
        return key_points, orientations, responses

    def extract_octave(self, image, key_points, orientations):
        mask = self.mask_border_key_points(image.shape, key_points, distance=20)
        key_points = key_points[mask]
        orientations = orientations[mask]
        descriptors = np.zeros((key_points.shape[0], self.positions.shape[0]), dtype=np.uint8)
        for i in range(descriptors.shape[0]):
            angle = orientations[i]
            sin_value = np.sin(angle)
            cos_value = np.cos(angle)
            key_point_row = key_points[i, 0]
            key_point_col = key_points[i, 1]
            for j in range(descriptors.shape[1]):
                position_row_0 = self.positions_0[j, 0]
                position_col_0 = self.positions_0[j, 1]
                position_row_1 = self.positions_1[j, 0]
                position_col_1 = self.positions_1[j, 1]
                rotated_position_row_0 = round(sin_value*position_row_0 + cos_value*position_col_0)
                rotated_position_col_0 = round(cos_value*position_row_0 - sin_value*position_col_0)
                rotated_position_row_1 = round(sin_value*position_row_1 + cos_value*position_col_1)
                rotated_position_col_1 = round(cos_value*position_row_1 - sin_value*position_col_1)
                if (image[key_point_row + rotated_position_row_0, key_point_col + rotated_position_col_0] <
                        image[key_point_row + rotated_position_row_1, key_point_col + rotated_position_col_1]):
                    descriptors[i, j] = 1
        return descriptors, mask

    def detect(self, image):
        pyramid = list(self.pyramid_gaussian(image))
        key_points_list = []
        responses_list = []
        scales_list = []
        orientations_list =[]
        descriptors_list = []
        for octave in range(len(pyramid)):
            octave_image = pyramid[octave]
            key_points, orientations, responses = self.detect_octave(octave_image)
            if len(key_points) == 0:
                key_points_list.append(key_points)
                responses_list.append(responses)
                descriptors_list.append(np.zeros((0, 256), dtype=bool))
                continue
            descriptors, mask = self.extract_octave(octave_image, key_points, orientations)
            scaled_key_points = key_points[mask] * self.down_scale ** octave
            key_points_list.append(scaled_key_points)
            responses_list.append(responses[mask])
            orientations_list.append(orientations[mask])
            scales_list.append(self.down_scale ** octave * np.ones(scaled_key_points.shape[0]))
            descriptors_list.append(descriptors)
        key_points = np.vstack(key_points_list)
        responses = np.hstack(responses_list)
        scales = np.hstack(scales_list)
        orientations = np.hstack(orientations_list)
        descriptors = np.vstack(descriptors_list)
        if key_points.shape[0] < self.n_key_points:
            self.key_points = key_points
            self.scales = scales
            self.orientations = orientations
            self.responses = responses
            self.descriptors = descriptors
        else:
            best_indices = responses.argsort()[::-1][:self.n_key_points]
            self.key_points = key_points[best_indices]
            self.scales = scales[best_indices]
            self.orientations = orientations[best_indices]
            self.responses = responses[best_indices]
            self.descriptors = descriptors[best_indices]


