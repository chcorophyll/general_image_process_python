"""
References:
https://zhuanlan.zhihu.com/p/144807771

"""
import numpy as np


class FindContours(object):

    def __init__(self):
        self.grid = np.array([[1, 1, 1, 1, 1, 1, 0, 0],
                              [1, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 0, 0]])
        self.reset()

    def reset(self):
        self.grid = np.pad(self.grid, ((1, 1), (1, 1)), "constant", constant_values=0)
        self.LNBD = 1
        self.NBD = 1
        self.display_with_number = True
        self.max_border_number = self.grid.shape[0] * self.grid.shape[1]
        self.contours_dict = {}
        self.contours_dict[1] = self.contour(-1, "Hole")

    def contour(self, parent, contour_type, start_point=None):
        if not start_point:
            start_point = [-1, -1]
        contour_result = {"parent": parent,
                          "contour_type": contour_type,
                          "son": [],
                          "start_point": start_point}
        return contour_result

    def trans_num_to_char(self, num):
        if self.display_with_number:
            return str(num)
        if num > 1:
            return chr(63+num)
        if num < 0:
            return chr(95-num)
        else:
            return str(num)

    def display_grid(self):
        for i in range(self.grid.shape[0]):
            num = '\033[0;37m' + '['
            print(num, end=' ')
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == 0:
                    num = '\033[0;37m' + self.trans_num_to_char(self.grid[i][j])
                    print(num, end=' ')
                else:
                    num = '\033[1;31m' + self.trans_num_to_char(self.grid[i][j])
                    print(num, end=' ')
            num = '\033[0;37m' + ']'
            print(num)
        print('\033[0;37m')

    def find_neighbor(self, center, start, clock_wise=1):
        if clock_wise == 1:
            weight = 1
        else:
            weight = -1
        neighbors = np.array([[0, 0], [0, 1], [0, 2], [1, 2],
                              [2, 2], [2, 1], [2, 0], [1, 0]])
        indexs = np.array([[0, 1, 2],
                           [7, 9, 3],
                           [6, 5, 4]])
        start_index = indexs[start[0]-center[0]+1][start[1]-center[1]+1]
        for i in range(1, len(neighbors)+1):
            current_index = (start_index + i*weight + 8) % 8
            x = neighbors[current_index][0] + center[0] - 1
            y = neighbors[current_index][1] + center[1] - 1
            if self.grid[x][y] != 0:
                return [x, y]
        return [-1, -1]

    def board_follow(self, center_point, start_point, mode):
        ij = center_point
        ij2 = start_point
        ij1 = self.find_neighbor(ij, ij2)
        x = ij1[0]
        y = ij1[1]
        if ij1 == [-1, -1]:
            self.grid[ij[0]][ij[1]] == -self.NBD
            return
        ij2 = ij1
        ij3 = ij
        for k in range(self.max_border_number):
            ij4 = self.find_neighbor(ij3, ij2, 0)
            x = ij3[0]
            y = ij3[1]
            if ij4[0] - ij2[0] <= 0:
                weight = -1
            else:
                weight = 1
            if self.grid[x][y] < 0:
                self.grid[x][y] = self.grid[x][y]
            elif self.grid[x][y-1] == 0 and self.grid[x][y+1] == 0:
                self.grid[x][y] = -self.NBD * weight
            elif self.grid[x][y+1] == 0:
                self.grid[x][y] = -self.NBD
            elif self.grid[x][y] == 1 and self.grid[x][y+1] != 0:
                self.grid[x][y] = self.NBD
            else:
                self.grid[x][y] = self.grid[x][y]
            if ij4 == ij and ij3 == ij1:
                return
            ij2 = ij3
            ij3 = ij4

    def raster_scan(self):
        for i in range(self.grid.shape[0]):
            self.LNBD = 1
            for j in range(self.grid.shape[1]):
                if abs(self.grid[i][j]) > 1:
                    self.LNBD = abs(self.grid[i][j])
                if self.grid[i][j] >= 1:
                    if self.grid[i][j] == 1 and self.grid[i][j-1] == 0:
                        self.NBD += 1
                        self.board_follow([i, j], [i, j-1], 1)
                        boarder_type = "Outer"
                    elif self.grid[i][j] > 1 and self.grid[i][j+1] == 0:
                        self.NBD += 1
                        self.board_follow([i, j], [i, j+1], 1)
                        boarder_type = "Hole"
                    else:
                        continue
                    parent = self.LNBD
                    if self.contours_dict[self.LNBD]["contour_type"] == boarder_type:
                        parent = self.contours_dict[self.LNBD]["parent"]
                    self.contours_dict[self.NBD] = self.contour(parent, boarder_type, [i-1, j-1])
                    self.contours_dict[parent]["son"].append(self.NBD)
        self.grid = self.grid[1:-1, 1:-1]
