"""
References:
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_texture.pyx#L95
"""
"""
Note: All edge modes implemented here follow the corresponding numpy.pad
conventions.
The table below illustrates the behavior for the array [1, 2, 3, 4], if padded
by 4 values on each side:
                               pad     original    pad
    constant (with c=0) :    0 0 0 0 | 1 2 3 4 | 0 0 0 0
    wrap                :    1 2 3 4 | 1 2 3 4 | 1 2 3 4
    symmetric           :    4 3 2 1 | 1 2 3 4 | 4 3 2 1
    edge                :    1 1 1 1 | 1 2 3 4 | 4 4 4 4
    reflect             :    3 4 3 2 | 1 2 3 4 | 3 2 1 2
"""
import numpy as np


def coord_map(dimensions, coordinate, mode):
    """
    Wrap a coordinate, according to a given mode.
    :param dimensions: int, maximum coordinate.
    :param coordinate: int, coordinate provided by user.  May be < 0 or > dimensions
    :param mode: {'W', 'S', 'R', 'E'}, Whether to wrap, symmetric reflect, reflect or use the nearest
        coordinate if `coord` falls outside [0, dim).
    :return: int, co
    """
    max_coordinate = dimensions - 1
    if mode == "S":
        if coordinate < 0:
            coordinate = -coordinate - 1
        if coordinate > max_coordinate:
            if (coordinate // dimensions) % 2 != 0:  # ?
                return max_coordinate - (coordinate % dimensions)
            else:
                return coordinate % dimensions
    elif mode == "W":
        if coordinate < 0:
            return max_coordinate - (-coordinate - 1) % dimensions
        if coordinate > max_coordinate:
            return coordinate % dimensions
    elif mode == "E":
        if coordinate < 0:
            return 0
        elif coordinate > max_coordinate:
            return max_coordinate
    elif mode == "R":
        if dimensions == 1:
            return 0
        elif coordinate < 0:
            if (-coordinate // max_coordinate) % 2 != 0:
                return max_coordinate - (-coordinate % max_coordinate)
            else:
                return -coordinate % max_coordinate
        elif coordinate > max_coordinate:
            if (coordinate // max_coordinate) % 2 != 0:
                return max_coordinate - (coordinate % max_coordinate)
            else:
                return coordinate % max_coordinate
    return coordinate


def get_pixel2d(image, rows, cols, row, col, mode="C", constant_value=0):
    if mode == "C":
        if (row < 0 or row >= rows) or (col < 0 or col >= cols):
            return constant_value
        else:
            return image[row, col]
    else:
        current_row = coord_map(rows, row, mode)
        current_col = coord_map(cols, col, mode)
        return image[current_row, current_col]


def _bilinear_interpolation(image, rows, cols, row, col, mode="C", constant_value=0):
    min_row = np.floor(row)
    min_col = np.floor(col)
    max_row = np.ceil(row)
    max_col = np.ceil(col)
    delta_row = row - min_row
    delta_col = col - min_col
    top_left = get_pixel2d(image, rows, cols, min_row, min_col, mode, constant_value)
    top_right = get_pixel2d(image, rows, cols, min_row, max_col, mode, constant_value)
    bottom_left = get_pixel2d(image, rows, cols, max_row, min_col, mode, constant_value)
    bottom_right = get_pixel2d(image, rows, cols, max_row, max_col, mode, constant_value)
    top = (1 - delta_col) * top_left + delta_col * top_right
    bottom = (1 - delta_col) * bottom_left + delta_col * bottom_right
    out = (1 - delta_row) * top + delta_row * bottom
    return out


def _bit_rotate_right(value, length):
    return (value >> 1) | ((value & 1) << (length - 1))


def local_binary_pattern(image, points_number, radius, method="D"):
    weights = 2 ** np.arange(points_number)
    rr = - radius * np.sin(2 * np.pi * np.arange(points_number) / points_number)
    cc = radius * np.cos(2 * np.pi * np.arange(points_number) / points_number)
    rp = np.round(rr, 5)
    cp = np.round(cc, 5)
    texture = np.zeros(points_number, dtype="float")
    signed_texture = np.zeros(points_number, dtype="int")
    rotation_chain = np.zeros(points_number, dtype="int")
    output_shape = (image.shape[0], image.shape[1])
    output = np.zeros(output_shape, dtype="float")
    rows = image.shape[0]
    cols = image.shape[1]
    for r in range(rows):
        for c in range(cols):
            for i in range(points_number):
                texture[i] = _bilinear_interpolation(image, rows, cols, r+rp[i], c+cp[i], "C", 0)
            for i in range(points_number):
                if texture[i] - image[r][c] >= 0:
                    signed_texture[i] = 1
                else:
                    signed_texture[i] = 0
            lbp = 0
            if method == "V":
                sum_ = 0
                var_ = 0
                for i in range(points_number):
                    sum_ += texture[i]
                    var_ += texture[i] * texture[i]
                lbp = (var_ - (sum_ * sum_) / points_number) / points_number  # var = E(x^2) - (E(x))^2
            elif method == "U" or method == "N":
                changes = 0
                for i in range(points_number-1):
                    changes += (signed_texture[i] - signed_texture[i+1]) != 0
                if method == "N":
                    if changes <= 2:
                        n_ones = 0
                        first_one = -1
                        first_zero = -1
                        for i in range(points_number):
                            if signed_texture[i]:
                                n_ones += 1
                                if first_one == -1:
                                    first_one = i
                            else:
                                if first_zero == -1:
                                    first_zero = i
                        if n_ones == 0:
                            lbp = 0
                        elif n_ones == points_number:
                            lbp = points_number * (points_number - 1) + 1
                        else:
                            if first_one == 0:
                                rot_index = n_ones - first_zero
                            else:
                                rot_index = points_number - first_one
                            lbp = 1 + (n_ones - 1) * points_number + rot_index  # 起点是xx00 ？？
                    else:
                        lbp = points_number * (points_number - 1) + 2
                else:
                    if changes <= 2:
                        for i in range(points_number):
                            lbp += signed_texture[i]
                    else:
                        lbp = points_number + 1
            else:
                # method == "default"
                for i in range(points_number):
                    lbp += signed_texture[i] * weights[i]
                if method == "R":
                    rotation_chain[0] = lbp
                    for i in range(1, points_number):
                        rotation_chain[i] = _bit_rotate_right(rotation_chain[i-1], points_number)
                    lbp = rotation_chain[0]
                    for i in range(1, points_number):
                        lbp = min(lbp, rotation_chain[i])
            output[r, c] = lbp
    return output






