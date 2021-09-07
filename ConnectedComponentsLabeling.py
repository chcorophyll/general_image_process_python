import numpy as np
from UnionFind import UnionFind


class ConnectedComponentsLabeling(object):

    def __init__(self, neighbor_number=4, max_components=100):
        self.neighbor_number = neighbor_number
        self.current_uf = UnionFind(max_components)

    def image_label(self, image_array):
        image_height, image_width = image_array.shape[0], image_array.shape[1]
        label_array = np.zeros((image_height, image_width))
        if self.neighbor_number == 4:
            neighbors = [(-1, 0), (0, -1)]
        else:
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
        label = 1
        # first pass
        for i in range(image_height):
            for j in range(image_width):
                if image_array[i][j] == 255:
                    level_label = []
                    for k in neighbors:
                        near_i = i + k[0]
                        near_j = j + k[1]
                        if 0 <= near_i < image_height and 0 <= near_j < image_width:
                            if image_array[near_i][near_j] == 255:
                                level_label.append(label_array[near_i][near_j])
                    if not level_label:
                        current_label = label
                        label += 1
                    else:
                        current_label = level_label.min()
                        for level in level_label:
                            if level != current_label:
                                self.current_uf.union(current_label, level)
                    label_array[i][j] = current_label
        # second pass
        for i in range(image_height):
            for j in range(image_width):
                if image_array[i][j] == 255:
                    label_array[i][j] = self.current_uf.find(label_array[i][j])
        return label_array, label - 1











