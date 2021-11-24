"""
References:
https://github.com/Simon-Hohberg/Viola-Jones/tree/master/violajones
https://medium.datadriveninvestor.com/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b
https://github.com/aparande/FaceDetection
"""
import numpy as np


def to_integral_image(image_array):
    row_sum = np.zeros(image_array.shape)
    integral_image_array = np.zeros((image_array.shape[0] + 1, image_array.shape[1] + 1))
    for x in range(image_array.shape[1]):
        for y in range(image_array.shape[0]):
            row_sum[y, x] = row_sum[y - 1, x] + image_array[y, x]
            integral_image_array[y + 1, x + 1] = integral_image_array[y + 1, x] + row_sum[y, x]
    return integral_image_array


def sum_region(integral_image_array, top_left, bottom_right):
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    if top_left == bottom_right:
        return integral_image_array[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return integral_image_array[bottom_right] - integral_image_array[top_right] - \
           integral_image_array[bottom_left] + integral_image_array[top_left]


def enum(**enums):
    return type("Enum", (), enums)


FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1),
                   THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
                FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


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
