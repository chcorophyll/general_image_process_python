"""
References:
https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC03%E7%AB%A0%20k%E8%BF%91%E9%82%BB%E6%B3%95/3.KNearestNeighbors.ipynb

"""
from collections import namedtuple
import numpy as np
import scipy.ndimage.filters

class KdNode(object):

    def __init__(self, element, split_axis, left=None, right=None):
        self.element = element
        self.split_axis = split_axis
        self.left = left
        self.right = right


class KdTree(object):

    def __init__(self, data):
        self.k = len(data[0])  # number of features
        self.first_split_axis = self.get_first_split_axis(data)
        self.root = self.create_node(data, self.first_split_axis)

    def get_first_split_axis(self, data_set):
        feature_variance = np.var(np.array(data_set), axis=0)
        first_spilt_axis = np.argmax(feature_variance)
        if 0 <= first_spilt_axis < self.k:
            return first_spilt_axis
        else:
            raise ValueError("Index not in Range Number of Features ")

    def create_node(self, data_set, split_axis):
        if not data_set:
            return None
        data_set.sort(key=lambda x: x[split_axis])
        split_index = len(data_set) // 2  # median
        median = data_set[split_index]
        next_split_axis = (split_axis + 1) % self.k
        left = self.create_node(data_set[:split_index], next_split_axis)
        right = self.create_node(data_set[split_index + 1:], next_split_axis)
        return KdNode(median, split_axis, left, right)


# Nearest Point
result = namedtuple("Result Tuple", "nearest_point nearest_distance visited_node_numbers")


def find_nearest(tree, target_point):
    k = len(target_point)

    def back_track(kd_node, target, max_distance):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)
        visited_node_numbers = 1
        split_axis = kd_node.split_axis
        pivot = kd_node.element
        if target[split_axis] <= pivot[split_axis]:
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left
        temp = back_track(nearer_node, target, max_distance)
        point = temp.nearest_point
        distance = temp.nearest_distance
        visited_node_numbers += temp.visited_node_numbers
        if distance < max_distance:
            max_distance = distance
        temp_distance = np.abs(pivot[split_axis] - target[split_axis])
        if max_distance < temp_distance:
            return result(point, distance, visited_node_numbers)

        temp_distance = np.sqrt(sum((p_1 - p_2) ** 2 for p_1, p_2 in zip(pivot, target)))
        if temp_distance < distance:
            point = pivot
            distance = temp_distance
            max_distance = distance
        temp_1 = back_track(further_node, target, max_distance)
        visited_node_numbers += temp_1.visited_node_numbers
        if temp_1.nearest_distance < distance:
            point = temp_1.nearest_point
            distance = temp_1.nearest_distance
        return result(point, distance, visited_node_numbers)

    return back_track(tree.root, target_point, float("inf"))


# K Nearest Point

def find_nearest(tree, target_point, number_near):

    def back_track(kd_node, target, max_distance):
        if kd_node is None:
            return
        split_axis = kd_node.split_axis
        pivot = kd_node.element
        if target[split_axis] <= pivot[split_axis]:
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left
        back_track(nearer_node, target, max_distance)
        result.sort(key=lambda x: x[1])
        if result[-1][1] < max_distance:
            max_distance = result[-1][1]
        temp_distance = np.abs(pivot[split_axis] - target[split_axis])
        if max_distance < temp_distance:
            return   # 球与超平面不相交
        temp_distance = np.sqrt(sum((p_1 - p_2) ** 2 for p_1, p_2 in zip(pivot, target)))
        if temp_distance < result[-1][1]:
            result[-1][0] = pivot
            result[-1][1] = temp_distance
            max_distance = temp_distance
        back_track(further_node, target, max_distance)

    k = len(target_point)
    result = [([0] * k, float("inf"))] * number_near
    back_track(tree.root, target_point, float("inf"))
    return result
