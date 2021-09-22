"""
References:
https://github.com/tbliu/FAST/blob/master/src/fast.py
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/corner_cy.pyx
"""
from skimage.feature import

def shape(array):
    rows = len(array)
    cols = len(array[0])
    return [rows, cols]


def zeros(rows, cols):
    return  [[0] * cols for i in range(rows)]


def rgb2gray(array):
    rows, cols = shape(array)
    gray_scale = zeros(rows, cols)
    for row in rows:
        for col in cols:
            red, green, blue = array[row][col]
            gray = int(0.3 * red + 0.59 * green + 0.11 * blue)
            gray_scale[row][col] = gray
    return gray


def median_blur(image, start_search_row, end_search_row, start_search_col, end_search_col, kernel_size=3):
    blur_img = image[:]
    for y in range(start_search_row, end_search_row):
        for x in range(start_search_col, end_search_col):
            current_window = []
            for j in range(y - kernel_size // 2, y + kernel_size // 2 + 1):
                for i in range(x - kernel_size // 2, x + kernel_size // 2 + 1):
                    current_window.append(blur_img[j][i])
            insert_sort(current_window)
            blur_img[y][x] = current_window[len(current_window) // 2]
    return blur_img


def insert_sort(window):
    for index in range(1, len(window)):
        current = window[index]
        position = index - 1
        while position >= 0 and window[position] > current:
            window[position + 1] = window[position]
            position -= 1
        window[position + 1] = current


def circle(row, col):
    point_1 = (row - 3, col)
    point_2 = (row - 3, col + 1)
    point_3 = (row - 2, col + 2)
    point_4 = (row - 1, col + 3)
    point_5 = (row, col + 3)
    point_6 = (row + 1, col + 3)
    point_7 = (row + 2, col + 2)
    point_8 = (row + 3, col + 1)
    point_9 = (row + 3, col)
    point_10 = (row + 3, col - 1)
    point_11 = (row + 2, col - 2)
    point_12 = (row + 1, col - 3)
    point_13 = (row, col - 3)
    point_14 = (row - 1, col - 3)
    point_15 = (row - 2, col - 2)
    point_16 = (row - 3, col - 1)
    return [point_1,point_2, point_3, point_4,
            point_5, point_6, point_7, point_8,
            point_9, point_10, point_11, point_12,
            point_13, point_14, point_15, point_16]


def is_corner(image, row, col, region_of_interest, threshold, n=12):
    intensity = int(image[row][col])
    row_1, col_1 = region_of_interest[0]
    row_9, col_9 = region_of_interest[8]
    row_5, col_5 = region_of_interest[4]
    row_13, col_13 = region_of_interest[12]
    intensity_1 = int(image[row_1][col_1])
    intensity_9 = int(image[row_9][col_9])
    intensity_5 = int(image[row_5][col_5])
    intensity_13 = int(image[row_13][col_13])
    four_check_list = [intensity_1, intensity_9, intensity_5, intensity_13]
    bright_count = 0
    dark_count = 0
    for check_intensity in four_check_list:
        if check_intensity - intensity > threshold:
            bright_count += 1
        elif intensity - check_intensity > threshold:
            dark_count += 1
    if bright_count < 3 and dark_count < 3:
        return False
    corner_type_list = [0] * 16
    for index, row, col in enumerate(region_of_interest):
        current_intensity = int(image[row][col])
        if current_intensity  - intensity > threshold:
            corner_type_list[index] = 1
        elif intensity - current_intensity  > threshold:
            corner_type_list[index] = -1
    # test bright
    consecutive_count = 0
    for index in range(15 + n):
        if corner_type_list[index % 16] == 1:
            consecutive_count += 1
            if consecutive_count == n:
                return True
        else:
            consecutive_count = 0
    # test
    consecutive_count = 0
    for index in range(15 + n):
        if corner_type_list[index % 16] == -1:
            consecutive_count += 1
            if consecutive_count == n:
                return True
        else:
            consecutive_count = 0
    return False


def is_adjacent(point_1, point_2):
    row_1, col_1 = point_1
    row_2, col_2 = point_2
    y_distance = (row_1 - row_2) ** 2
    x_distance = (col_1 - col_2) ** 2
    return (y_distance + x_distance) ** 0.5 <= 4


def calculate_score(image, point, region_of_interest):
     intensity = int(image[point[0]][point[1]])
     neighbor_intensity = [abs(intensity - int(image[row][col])) for row, col in region_of_interest]
     return sum(neighbor_intensity)


def suppress(image, corners, regions_of_interest):
    i = 1
    while i < len(corners):
        current_point = corners[i]
        previous_point = corners[i - 1]
        if is_adjacent(previous_point, current_point):
            current_score = calculate_score(image, current_point, regions_of_interest[i])
            previous_score = calculate_score(image, previous_point, regions_of_interest[i-1])
            if current_score > previous_score:
                del(corners[i - 1])
                del(regions_of_interest[i - 1])
            else:
                del(corners[i])
                del(regions_of_interest[i])
            i -= 1
        i += 1
    return corners


def detect(image, threshold=50):
    image = rgb2gray(image)
    corners = []
    regions_of_interest = []
    rows, cols = shape(image)
    start_search_row = int(0.25 * rows)
    end_search_row = int(0.75 * rows)
    start_search_col = int(0.25 * cols)
    end_search_col = int(0.75 * cols)
    image = median_blur(image, start_search_row, end_search_row, start_search_col, end_search_col)

    for row in range(start_search_row, end_search_row):
        for col in range(start_search_col, end_search_col):
            region_of_interest = circle(row, col)
            if is_corner(image, row, col, region_of_interest, threshold):
                corners.append((row, col))
                regions_of_interest.append(region_of_interest)
    corners = suppress(image, corners, regions_of_interest)
    return corners