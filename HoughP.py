"""
References:
https://gist.github.com/Socret360/e01e8916c439aa73c8a437592b41c252#file-line-detection-with-hough-transform-non-vectorized-py
https://blog.csdn.net/zhaocj/article/details/40047397

"""
import random
import cv2
import numpy as np


def line_detection_non_vectorized(edge_image, num_rhos=180, num_thetas=180,
                                  t_count=220, max_line_gap=100, min_line_length=100, max_line_number=100):
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    accumulator = np.zeros((len(rhos), len(rhos)))
    mask = np.zeros((edge_height, edge_width))
    points = []
    edge_lines = []
    # collect edge points
    for y in range(edge_height):
        for x in range(edge_width):
            if edge_image[y][x] != 0:
                mask[y][x] = 1
                points.append((y, x))
    edge_points_count = len(points)
    while edge_points_count > 0:
        idx = random.randint(0, edge_points_count - 1)
        point_y, point_x = points[idx]
        points[idx] = points[-1]
        if not mask[point_y][point_x]:
            continue
        max_point_accumulator = t_count - 1
        max_point_theta_index = 0
        for theta_idx in range(len(thetas)):
            edge_point = [point_y - edge_height_half, point_x - edge_width_half]
            rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
            theta = thetas[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx][theta_idx] += 1
            if max_point_accumulator < accumulator[rho_idx][theta_idx]:
                max_point_accumulator = accumulator[rho_idx][theta_idx]
                max_point_theta_index = theta_idx
        if max_point_accumulator < t_count:
            continue
        # move align line direction
        sin_theta = -sin_thetas[max_point_theta_index]
        cos_theta = cos_thetas[max_point_theta_index]
        x_0 = point_x
        y_0 = point_y
        # theta：hough angle， beta：line slope angle
        # case 1: 0 < theta < 1/2 pi -> 1/2 pi < beta < pi
        # tan(beta) = tan(pi - (1/2 pi - theta)) = tan(1/2 pi + theta) = sin(1/2 pi + theta) / cos (1/2 pi + theta)
        # tan(beta) = (sin(1/2 pi)*cos(theta) + cos(1/2 pi)sin(theta))/(cos(1/2 pi)cos(theta) - sin(1/2 pi)sin(theta))
        # tan(beta) = - cos(theta) / sin(theta)
        # case 1.1:  0 < theta < 1/4 pi -> 1/2 pi < beta < 3/4 pi -> line close to y axis
        # case 1.2:  1/4 pi < theta < 1/2 pi -> 3/4 pi < beta < pi -> line close to x axis
        # case 2 : #  1/2 pi < theta < pi -> 0 < beta < 1/2 pi
        # tan(beta) = tan(1/2 pi - (pi - theta)) = tan(theta - 1/2 pi) = sin(theta - 1/2 pi) / cos(theta - 1/2 pi)
        # tan(beta) = (sin(theta)cos(1/2 pi) - cos(theta)sin(1/2 pi))/(cos(theta)cos(1/2 pi)+sin(theta)sin(1/2 pi))
        # tan(beta) = - cos(theta) / sin(theta)
        # case 2.1: 1/2 pi < theta < 3/4 pi -> 0 < beta < 1/4 pi -> line close to x axis
        # case 2.2: 3/4 pi < theta < pi -> 1/4 pi < beta < 1/2 pi -> line close to y axis
        # conclusion : 1/4 pi < theta < 3/4 pi -> line close to x axis
        # conclusion : else  -> line close to y axis
        # 1/4 pi < theta < 3/4 pi -> line close to x axis
        if abs(sin_theta) > abs(cos_theta):
            x_flag = True
            if sin_theta > 0:
                dx = 1
            else:
                dx = -1
            dy = round(cos_theta / abs(sin_theta))
        # 0< theta < 1/4 pi or 3/4 pi < theta < pi  -> line close to y axis
        else:
            x_flag = False
            if cos_theta > 0:
                dy = 1
            else:
                dy = -1
            dx = round(sin_theta / abs(cos_theta))
        # find two points
        line_points = [(0, 0), (0, 0)]
        for k in range(2):
            # opposite direction
            current_x = x_0
            current_y = y_0
            current_gap = 0
            current_dx = dx
            current_dy = dy
            if k > 0:
                current_dx = -dx
                current_dy = -dy
            while True:
                if current_x < 0 or current_x >= edge_width or current_y < 0 or current_y >= edge_height:
                    break
                if x_flag:
                    current_x = current_x + dx
                    current_y = current_y + dy
                else:
                    current_y = current_y + dy
                    current_x = current_x + dx
                if mask[current_y][current_x]:
                    current_gap = 0
                    line_points[k] = (current_y, current_x)
                else:
                    current_gap += 1
                    if current_gap > max_line_gap:
                        break
        if abs(line_points[0][0] - line_points[1][0]) >= min_line_length \
                and abs(line_points[0][1] - line_points[1][1]) >= min_line_length:
            good_line = True
        else:
            good_line = False
        # second search for accumulator
        for k in range(2):
            # opposite direction
            current_x = x_0
            current_y = y_0
            current_dx = dx
            current_dy = dy
            if k > 0:
                current_dx = -dx
                current_dy = -dy
            while True:
                if current_x < 0 or current_x >= edge_width or current_y < 0 or current_y >= edge_height:
                    break
                if x_flag:
                    current_x = current_x + dx
                    current_y = current_y + dy
                else:
                    current_y = current_y + dy
                    current_x = current_x + dx
                if mask[current_y][current_x]:
                    if good_line:
                        for theta_idx in range(len(thetas)):
                            edge_point = [point_y - edge_height_half, point_x - edge_width_half]
                            rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
                            theta = thetas[theta_idx]
                            rho_idx = np.argmin(np.abs(rhos - rho))
                            accumulator[rho_idx][theta_idx] -= 1
                        mask[current_y][current_x] = 0
                if current_y == line_points[0][0] and current_x == line_points[0][1]:
                    break
        if good_line:
            edge_lines.append([line_points[0][0], line_points[0][1], line_points[1][0], line_points[1][1]])
            if len(edge_lines) >= max_line_number:
                return edge_lines
    return edge_lines


if __name__ == "__main__":
    for i in range(3):
        image = cv2.imread(f"sample-{i + 1}.png")
        edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
        edge_image = cv2.Canny(edge_image, 100, 200)
        edge_image = cv2.dilate(
            edge_image,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1
        )
        edge_image = cv2.erode(
            edge_image,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1
        )
        line_detection_non_vectorized(image, edge_image)
