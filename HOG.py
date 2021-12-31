"""
References:
https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py
https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
https://gist.github.com/MrinalTyagi/7f39818967067e82e3e9d01ed31b817d#file-hog-ipynb
"""
import numpy as np


def _hog_channel_gradient(channel):
    g_row = np.empty(channel, dtype=channel.type)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel, dtype=channel.type)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] -channel[:, :-2]
    return g_row, g_col


def cell_hog(magnitude, orientation, orientation_start, orientation_end,
             cell_cols, cell_rows, col_index, row_index, size_cols, size_rows,
             range_rows_start, range_rows_stop, range_cols_start, range_cols_stop):
    total = 0
    for cell_row in range(range_rows_start, range_rows_stop):
        cell_row_index = cell_row + row_index
        if cell_row_index < 0 or cell_row_index >= size_rows:
            continue
        for cell_col in range(range_cols_start, range_cols_stop):
            cell_col_index = cell_col + col_index
            if cell_col_index < 0 or cell_col_index >= size_cols:
                continue
            if orientation[cell_row_index, cell_col_index] >= orientation_start:
                continue
            if orientation[cell_row_index, cell_col_index] < orientation_end:
                continue
            total += magnitude[cell_row_index, cell_col_index]
    return total / (cell_rows * cell_cols)  # 与原论文不同 原论文按照距离插值


def _hog_histograms(gradient_columns, gradient_rows, cell_columns, cell_rows,
                    size_columns, size_rows, num_of_cell_cols, num_of_cell_rows,
                    num_of_orientations, orientation_histogram):
    magnitude = np.hypot(gradient_columns, gradient_rows)
    orientation = np.rad2deg(np.arctan2(gradient_rows, gradient_columns)) % 180  # 与论文不一样 论文是取绝对值 而不是180取余
    r_0 = cell_rows // 2
    c_0 = cell_columns // 2
    cc = cell_rows * num_of_cell_rows
    cr = cell_columns * num_of_cell_cols
    range_rows_stop = (cell_rows + 1) // 2
    range_rows_start = - cell_rows // 2
    range_columns_stop = (cell_columns + 1) // 2
    range_columns_start = - cell_columns // 2
    num_of_orientations_per_180 = 180 / num_of_orientations
    # 以中心点平移
    while True:
        for i in range(num_of_orientations):
            orientation_start = num_of_orientations_per_180 * (i + 1)
            orientation_stop = num_of_orientations_per_180 * i
            c = c_0
            r = r_0
            r_i = 0
            c_i = 0
            while r < cc:
                c_i = 0
                c = c_0
                while c < cr:
                    orientation_histogram[r_i, c_i, i] = cell_hog(magnitude, orientation,
                                                                  orientation_start, orientation_stop,
                                                                  cell_columns, cell_rows,
                                                                  c, r,
                                                                  size_columns, size_rows,
                                                                  range_rows_start, range_rows_stop,
                                                                  range_columns_start, range_columns_stop)
                    c_i += 1
                    c += cell_columns
                r_i += 1
                r += cell_rows


def _hog_normalized_block(block, method, eps=1e-5):
    if method == "L1":
        out = block / (np.sum(np.abs(block)) + eps)
    elif method == "L1-sqrt":
        out = np.sqrt(block / (np.sum(np.abs(block)) + eps))
    elif method == "L2":
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
    elif method == "L2-Hys":
        out = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        out = np.minimum(out, 0.2)
        out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    else:
        raise ValueError('Selected block normalization method is invalid.')
    return out


def hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
        block_norm="L2-Hys", transform_sqrt=False, feature_vector=True, multichannel=None, channel_axis=None):
    image = np.atleast_2d(image)
    image = image.astype("float", copy=False)
    ndim_spatial = image.ndim - 1 if multichannel else image.ndim
    if ndim_spatial != 2:
        raise ValueError('Only images with two spatial dimensions are '
                         'supported. If using with color/multichannel '
                         'images, specify `channel_axis`.')
    # 1 图像预处理
    # gamma
    if transform_sqrt:
        image = np.sqrt(image)
    # 2 图像梯度
    if multichannel:
        g_row_by_ch = np.empty_like(image, dtype="float")
        g_col_by_ch = np.empty_like(image, dtype="float")
        g_magnitude = np.empty_like(image, dtype="float")
        for idx_ch in range(image.shape[2]):
            g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = _hog_channel_gradient(image[:, :, idx_ch])
            g_magnitude[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch])
        idcs_max = g_magnitude.argmax(axis=2)
        rr, cc = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij", sparse=True)
        g_row = g_row_by_ch[rr, cc, idcs_max]
        g_col = g_col_by_ch[rr, cc, idcs_max]
    else:
        g_row, g_col = _hog_channel_gradient(image)
    # 3 cell直方图生成
    s_row, s_col = image.shape[:2]  # 高 宽
    c_row, c_col = pixels_per_cell  # 8, 8
    b_row, b_col = cells_per_block  # 3, 3
    n_cells_row = int(s_row // c_row)  # 高 cell 数目
    n_cells_col = int(s_col // c_col)  # 宽 cell 数目
    orientation_histogram = np.zeros((n_cells_row, n_cells_col, orientations), dtype="float")
    g_row = g_row.astype("float", copy=False)
    g_col = g_col.astype("float", copy=False)
    _hog_histograms(g_col, g_row, c_col, c_row, s_col, s_row,
                    n_cells_col, n_cells_row, orientations, orientation_histogram)
    # 4 block直方图生成
    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    normalized_blocks = np.zeros(n_blocks_row, n_blocks_col, b_row, b_col, orientations, dtype="float")
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = orientation_histogram[r: r+b_row, c: c+c_col]
            normalized_blocks[r, c, :] = _hog_normalized_block(block, method=block_norm)
    if feature_vector:
        normalized_blocks = normalized_blocks.ravel()
    return normalized_blocks