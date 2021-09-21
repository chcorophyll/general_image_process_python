"""
References:
https://github.com/tbliu/FAST/blob/master/src/fast.py

"""


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
    point_3 = (row - 2, col + 2)
    point_5 = (row, col + 3)
    point_7 = (row + 2, col + 2)
    point_9 = (row + 3, col)
    point_11 = (row + 2, col - 2)
    point_13 = (row, col - 3)
    point_15 = (row - 2, col - 2)
    return [point_1, point_3, point_5, point_7, point_9, point_11, point_13, point_15]


def is_corner(image, row, col, region_of_interest, threshold):
    intensity = int(image[row][col])
    row_1, col_1 = region_of_interest[0]
    row_9, col_9 = region_of_interest[4]
    row_5, col_5 = region_of_interest[2]
    row_13, col_13 = region_of_interest[6]
    intensity_1 = int(image[row_1][col_1])
    intensity_9 = int(image[row_9][col_9])
    intensity_5 = int(image[row_5][col_5])
    intensity_13 = int(image[row_13][col_13])
    flag_1 = 0
    flag_9 = 0
    if abs(intensity_1 - intensity) > threshold:
        if intensity_1 - intensity > 0:
            flag_1 = 1
        else:
            flag_1 = -1
    if abs(intensity_9 - intensity) > threshold:
        if intensity_9 - intensity > 0:
            flag_9 = 1
        else:
            flag_9 = -1
    flag_first = flag_1 * flag_9
    if flag_first != 1:
        return False
    else:
        if abs(intensity_5 - intensity) > threshold:
            if intensity_5 - intensity > 0 and flag_9 == 1:
                return True
            elif intensity_5 - intensity < 0 and flag_9 == -1:
                return True
        if abs(intensity_13 - intensity) > threshold:
            if intensity_13 - intensity > 0 and flag_9 == 1:
                return True
            elif intensity_13 - intensity < 0 and flag_9 == -1:
                return True
        return False


def is_adjacent(point_1, point_2):
    row_1, col_1 = point_1
    row_2, col_2 = point_2
    y_distance = (row_1 - row_2) ** 2
    x_distance = (col_1 - col_2) ** 2
    return (y_distance + x_distance) ** 0.5 <= 4


def calculate_score(image, point, region_of_interest):
    row, col = point
    intensity = int(image[row][col])
    row_1, col_1 = region_of_interest[0]
    intensity_1 = int(image[row_1][col_1])
    row_3, col_3 = region_of_interest[1]
    intensity_3 = int(image[row_3][col_3])
    row_5, col_5 = region_of_interest[2]
    intensity_5 = int(image[row_5][col_5])
    row_7, col_7 = region_of_interest[3]
    intensity_7 = int(image[row_7][col_7])
    row_9, col_9 = region_of_interest[4]
    intensity_9 = int(image[row_9][col_9])
    row_11, col_11 = region_of_interest[5]
    intensity_11 = int(image[row_11][col_11])
    row_13, col_13 = region_of_interest[6]
    intensity_13 = int(image[row_13][col_13])
    row_15, col_15 = region_of_interest[7]
    intensity_15 = int(image[row_15][col_15])
    score = abs(intensity_1 - intensity) + abs(intensity_3 - intensity) + \
            abs(intensity_5 - intensity) + abs(intensity_7 - intensity) + \
            abs(intensity_9 - intensity) + abs(intensity_11 - intensity) + \
            abs(intensity_13 - intensity) + abs(intensity_15 - intensity)
    return score


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