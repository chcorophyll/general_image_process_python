"""
References:
https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC03%E7%AB%A0%20k%E8%BF%91%E9%82%BB%E6%B3%95/3.KNearestNeighbors.ipynb

"""
from collections import namedtuple
import numpy as np


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
class Node(namedtuple("Node", "location split_axis left_child right_child")):

    def __repr__(self):
        return str(tuple(self))


class Kdtree():

    def __init__(self, k=1):
        self.k = k
        self.kdtree = None

    def get_first_split_axis(self, data_set):
        feature_variance = np.var(np.array(data_set), axis=0)
        first_spilt_axis = np.argmax(feature_variance)
        if 0 <= first_spilt_axis < self.k:
            return first_spilt_axis
        else:
            raise ValueError("Index not in Range Number of Features ")

    def _fit(self, data_set, depth=None):
        try:
            k = self.k
        except IndexError as e:
            return None
        if not depth:
            first_split_axis = self.get_first_split_axis(data_set) % k
        else:
            first_split_axis = depth % k
        data_set.sort(key=lambda x: x[first_split_axis])
        median = len(data_set) // 2
        try:
            data_set[median]
        except IndexError:
            return None
        return Node(location=data_set[median],
                    split_axis=first_split_axis,
                    left_child=self._fit(data_set[:median], (first_split_axis+1) % k),
                    right_child=self._fit(data_set[median+1:], (first_split_axis+1) % k))

    def _search(self, target_point, tree=None, depth=None, best=None):
        if tree is None:
            return best
        k = self.k
        if not depth:
            first_split_axis = tree.split_axis % k
        else:
            first_split_axis = depth % k
        if target_point[first_split_axis] < tree.location[first_split_axis]:
            next_node = tree.left_child
        else:
            next_node = tree.right_child
        if next_node:
            best = next_node.location
        return self._search(target_point, tree=next_node, depth=(first_split_axis+1) % k, best=best)

    def fit(self, data_set):
        self.kdtree = self._fit(data_set)
        return self.kdtree

    def predict(self, point):
        result = self._search(point, self.kdtree)
        return result



