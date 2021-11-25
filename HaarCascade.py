"""
References:
https://github.com/Simon-Hohberg/Viola-Jones/tree/master/violajones
https://medium.datadriveninvestor.com/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b
https://github.com/aparande/FaceDetection
"""
from multiprocessing import Pool
from functools import partial
import progressbar
import numpy as np


LOADING_BAR_LENGTH = 50


# construct integral image
def to_integral_image(image_array):
    row_sum = np.zeros(image_array.shape)
    integral_image_array = np.zeros((image_array.shape[0] + 1, image_array.shape[1] + 1))
    for x in range(image_array.shape[1]):
        for y in range(image_array.shape[0]):
            row_sum[y, x] = row_sum[y - 1, x] + image_array[y, x]
            integral_image_array[y + 1, x + 1] = integral_image_array[y + 1, x] + row_sum[y, x]
    return integral_image_array


# integral compute
def sum_region(integral_image_array, top_left, bottom_right):
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    if top_left == bottom_right:
        return integral_image_array[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return integral_image_array[bottom_right] - integral_image_array[top_right] - \
           integral_image_array[bottom_left] + integral_image_array[top_left]


# enum type
def enum(**enums):
    return type("Enum", (), enums)


FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1),
                   THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
                FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


# Haar Feature
class HaarLikeFeature(object):

    def __init__(self, feature_type, position, width, height, threshold, polarity):
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = 1

    def get_score(self, integral_image):
        score = 0
        if self.type == FeatureType.TWO_VEWRTICAL:
            first = sum_region(integral_image,
                               self.top_left,
                               (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
            second = sum_region(integral_image,
                                (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                                self.bottom_right)
            score = first - second
        elif self.type == FeatureType.TWO_HORIZONTAL:
            first = sum_region(integral_image,
                               self.top_left,
                               (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
            second = sum_region(integral_image,
                                (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                self.bottom_right)
            score = first - second
        elif self.type == FeatureType.THREE_HORIZONTAL:
            first = sum_region(integral_image,
                               self.top_left,
                               (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
            second = sum_region(integral_image,
                                (int(self.top_left[0] + self.width / 3), self.top_left[1]),
                                (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
            third = sum_region(integral_image,
                               (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]),
                               self.bottom_right)
            score =  first - second + third
        elif self.type == FeatureType.THREE_VERTICAL:
            first = sum_region(integral_image,
                               self.top_left,
                               (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
            second = sum_region(integral_image,
                                (self.top_left[0], int(self.top_left[1] + self.height / 3)),
                                (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))

            third = sum_region(integral_image,
                               (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)),
                               self.bottom_right)
            score = first - second + third
        elif self.type == FeatureType.FOUR:
            first = sum_region(integral_image,
                               self.top_left,
                               (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
            second = sum_region(integral_image,
                                (int(self.top_left[0] + self.width / 2), self.top_left[1]),
                                (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
            third = sum_region(integral_image,
                               (self.top_left[0], int(self.top_left[1] + self.height / 2)),
                               (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
            fourth = sum_region(integral_image,
                                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)),
                                self.bottom_right)
            score = first - second - third + fourth
        return score

    def get_vote(self, integral_image):
        score = self.get_score(integral_image)
        return self.weight * (1 if score < self.polarity * self.threshold else -1)


# create feature
def create_features(img_height, img_width, min_feature_width,
                    max_feature_width, min_feature_height, max_feature_height):
    features = []
    for feature in FeatureTypes:
        feature_start_width = max(min_feature_width, feature[0])
        for feature_width in range(feature_start_width, max_feature_width, feature[0]):
            feature_start_height = max(min_feature_height, feature[1])
            for feature_height in range(feature_start_height, max_feature_height, feature[1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, 1))
                        features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, -1))
    return features


# get feature vote
def get_feature_vote(feature, image):
    return feature.get_vote(image)


# adaboost learn
def learn(positive_integral_images, negative_integral_images,
          num_classifiers=-1, min_feature_width=1,
          max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
    num_pos = len(positive_integral_images)
    num_neg = len(negative_integral_images)
    num_imgs = num_pos + num_neg
    img_height, img_width = positive_integral_images[0].shape
    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if min_feature_width == -1 else max_feature_width
    positive_weights = np.ones(num_pos) * 1.0 / (2 * num_pos)
    negative_weights = np.ones(num_neg) * 1.0 / (2 * num_neg)
    weights = np.hstack((positive_weights, negative_weights))
    labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))
    images = positive_integral_images + negative_integral_images
    features = create_features(img_height, img_width, min_feature_width,
                               max_feature_width, min_feature_height, max_feature_width)
    num_features = len(features)
    feature_indexs = list(range(num_features))
    num_classifiers = num_features if num_classifiers == -1 else num_classifiers
    votes = np.zeros((num_imgs, num_features))
    bar = progressbar.ProgressBar()
    pool = Pool(processes=None)
    for i in bar(range(num_imgs)):
        votes[i, :] = np.array(list(pool.map(partial(get_feature_vote, image=images[i]), features)))
    classifiers = []
    bar = progressbar.ProgressBar()
    for _ in bar((range(num_classifiers))):
        classification_errors = np.zeros(len(feature_indexs))
        weights *= 1 / np.sum(weights)
        for f in range(len(feature_indexs)):
            f_idx = feature_indexs[f]
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0,
                            range(num_imgs)))
            classification_errors[f] = error
        min_error_idx = np.argmin(classification_errors)
        best_error = classification_errors[min_error_idx]
        best_feature_idx = feature_indexs[min_error_idx]
        best_feature = features[best_feature_idx]
        feature_weight = 0.5 * np.log((1 - best_error) / best_error)
        best_feature.weight = feature_weight
        classifiers.append(best_feature)
        weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error)
        if labels[img_idx] != votes[img_idx, best_feature_idx]
        else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs))))
        feature_indexs.remove(best_feature_idx)
    return classifiers



