"""
References:
https://github.com/Simon-Hohberg/Viola-Jones/tree/master/violajones
https://medium.datadriveninvestor.com/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b
https://github.com/aparande/FaceDetection
"""
import math
import pickle
from multiprocessing import Pool
from functools import partial
import progressbar
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif

# LOADING_BAR_LENGTH = 50
#
#
# # construct integral image
# def to_integral_image(image_array):
#     row_sum = np.zeros(image_array.shape)
#     integral_image_array = np.zeros((image_array.shape[0] + 1, image_array.shape[1] + 1))
#     for x in range(image_array.shape[1]):
#         for y in range(image_array.shape[0]):
#             row_sum[y, x] = row_sum[y - 1, x] + image_array[y, x]
#             integral_image_array[y + 1, x + 1] = integral_image_array[y + 1, x] + row_sum[y, x]
#     return integral_image_array
#
#
# # integral compute
# def sum_region(integral_image_array, top_left, bottom_right):
#     top_left = (top_left[1], top_left[0])
#     bottom_right = (bottom_right[1], bottom_right[0])
#     if top_left == bottom_right:
#         return integral_image_array[top_left]
#     top_right = (bottom_right[0], top_left[1])
#     bottom_left = (top_left[0], bottom_right[1])
#     return integral_image_array[bottom_right] - integral_image_array[top_right] - \
#            integral_image_array[bottom_left] + integral_image_array[top_left]
#
#
# # enum type
# def enum(**enums):
#     return type("Enum", (), enums)
#
#
# FeatureType = enum(TWO_VERTICAL=(1, 2), TWO_HORIZONTAL=(2, 1),
#                    THREE_HORIZONTAL=(3, 1), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
# FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL, FeatureType.THREE_VERTICAL,
#                 FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]
#
#
# # Haar Feature
# class HaarLikeFeature(object):
#
#     def __init__(self, feature_type, position, width, height, threshold, polarity):
#         self.type = feature_type
#         self.top_left = position
#         self.bottom_right = (position[0] + width, position[1] + height)
#         self.width = width
#         self.height = height
#         self.threshold = threshold
#         self.polarity = polarity
#         self.weight = 1
#
#     def get_score(self, integral_image):
#         score = 0
#         if self.type == FeatureType.TWO_VEWRTICAL:
#             first = sum_region(integral_image,
#                                self.top_left,
#                                (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))
#             second = sum_region(integral_image,
#                                 (self.top_left[0], int(self.top_left[1] + self.height / 2)),
#                                 self.bottom_right)
#             score = first - second
#         elif self.type == FeatureType.TWO_HORIZONTAL:
#             first = sum_region(integral_image,
#                                self.top_left,
#                                (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))
#             second = sum_region(integral_image,
#                                 (int(self.top_left[0] + self.width / 2), self.top_left[1]),
#                                 self.bottom_right)
#             score = first - second
#         elif self.type == FeatureType.THREE_HORIZONTAL:
#             first = sum_region(integral_image,
#                                self.top_left,
#                                (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))
#             second = sum_region(integral_image,
#                                 (int(self.top_left[0] + self.width / 3), self.top_left[1]),
#                                 (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height))
#             third = sum_region(integral_image,
#                                (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]),
#                                self.bottom_right)
#             score =  first - second + third
#         elif self.type == FeatureType.THREE_VERTICAL:
#             first = sum_region(integral_image,
#                                self.top_left,
#                                (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))
#             second = sum_region(integral_image,
#                                 (self.top_left[0], int(self.top_left[1] + self.height / 3)),
#                                 (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3)))
#
#             third = sum_region(integral_image,
#                                (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)),
#                                self.bottom_right)
#             score = first - second + third
#         elif self.type == FeatureType.FOUR:
#             first = sum_region(integral_image,
#                                self.top_left,
#                                (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)))
#             second = sum_region(integral_image,
#                                 (int(self.top_left[0] + self.width / 2), self.top_left[1]),
#                                 (self.bottom_right[0], int(self.top_left[1] + self.height / 2)))
#             third = sum_region(integral_image,
#                                (self.top_left[0], int(self.top_left[1] + self.height / 2)),
#                                (int(self.top_left[0] + self.width / 2), self.bottom_right[1]))
#             fourth = sum_region(integral_image,
#                                 (int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2)),
#                                 self.bottom_right)
#             score = first - second - third + fourth
#         return score
#
#     def get_vote(self, integral_image):
#         score = self.get_score(integral_image)
#         return self.weight * (1 if score < self.polarity * self.threshold else -1)
#
#
# # create feature
# def create_features(img_height, img_width, min_feature_width,
#                     max_feature_width, min_feature_height, max_feature_height):
#     features = []
#     for feature in FeatureTypes:
#         feature_start_width = max(min_feature_width, feature[0])
#         for feature_width in range(feature_start_width, max_feature_width, feature[0]):
#             feature_start_height = max(min_feature_height, feature[1])
#             for feature_height in range(feature_start_height, max_feature_height, feature[1]):
#                 for x in range(img_width - feature_width):
#                     for y in range(img_height - feature_height):
#                         features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, 1))  ## ???
#                         features.append(HaarLikeFeature(feature, (x, y), feature_width, feature_height, 0, -1))
#     return features
#
#
# # get feature vote
# def get_feature_vote(feature, image):
#     return feature.get_vote(image)
#
#
# # adaboost learn
# def learn(positive_integral_images, negative_integral_images,
#           num_classifiers=-1, min_feature_width=1,
#           max_feature_width=-1, min_feature_height=1, max_feature_height=-1):
#     num_pos = len(positive_integral_images)
#     num_neg = len(negative_integral_images)
#     num_imgs = num_pos + num_neg
#     img_height, img_width = positive_integral_images[0].shape
#     max_feature_height = img_height if max_feature_height == -1 else max_feature_height
#     max_feature_width = img_width if min_feature_width == -1 else max_feature_width
#     positive_weights = np.ones(num_pos) * 1.0 / (2 * num_pos)  # positive_probability(1/2) * positive_sample_probability(1 / num_pos)
#     negative_weights = np.ones(num_neg) * 1.0 / (2 * num_neg)
#     weights = np.hstack((positive_weights, negative_weights))
#     labels = np.hstack((np.ones(num_pos), np.ones(num_neg) * -1))
#     images = positive_integral_images + negative_integral_images
#     features = create_features(img_height, img_width, min_feature_width,
#                                max_feature_width, min_feature_height, max_feature_width)
#     num_features = len(features)
#     feature_indexs = list(range(num_features))
#     num_classifiers = num_features if num_classifiers == -1 else num_classifiers
#     votes = np.zeros((num_imgs, num_features))
#     bar = progressbar.ProgressBar()
#     pool = Pool(processes=None)
#     for i in bar(range(num_imgs)):
#         votes[i, :] = np.array(list(pool.map(partial(get_feature_vote, image=images[i]), features)))
#     classifiers = []
#     bar = progressbar.ProgressBar()
#     for _ in bar((range(num_classifiers))):
#         classification_errors = np.zeros(len(feature_indexs))
#         weights *= 1 / np.sum(weights)
#         for f in range(len(feature_indexs)):
#             f_idx = feature_indexs[f]
#             error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != votes[img_idx, f_idx] else 0,
#                             range(num_imgs)))
#             classification_errors[f] = error
#         min_error_idx = np.argmin(classification_errors)
#         best_error = classification_errors[min_error_idx]
#         best_feature_idx = feature_indexs[min_error_idx]
#         best_feature = features[best_feature_idx]
#         feature_weight = 0.5 * np.log((1 - best_error) / best_error)
#         best_feature.weight = feature_weight
#         classifiers.append(best_feature)
#         weights = np.array(list(map(lambda img_idx: weights[img_idx] * np.sqrt((1-best_error)/best_error)
#         if labels[img_idx] != votes[img_idx, best_feature_idx]
#         else weights[img_idx] * np.sqrt(best_error/(1-best_error)), range(num_imgs))))
#         feature_indexs.remove(best_feature_idx)
#     return classifiers


# version above not clear on adaboost
def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(image.shape[0]):
        for x in range(image.shape[0]):
            s[y][x] = s[y-1][x] + image[y][x] if y - 1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1] + s[y][x] if x - 1 >= 0 else s[y][x]
    return ii


class RectangleRegion:

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] \
               - ii[self.y+self.height][self.x] - ii[self.y][self.x+self.width]

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)

    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)


class WeakClassifier:

    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions])
        - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s, %s)" % (self.threshold,
                                                                 self.polarity,
                                                                 str(self.positive_regions),
                                                                 str(self.negative_regions))


class ViolaJones:

    def __init__(self, T=10):
        self.T = T
        self.alphas = []
        self.clfs = []

    def build_features(self, image_shape):
        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width:
                            features.append(([right], [immediate]))
                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height:
                            features.append(([immediate], [bottom]))
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        if i + 3 * w < width:
                            features.append(([right], [right_2, immediate]))
                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height:
                            features.append(([bottom], [bottom_2, immediate]))
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))
                        j += 1
                i += 1
        return np.array(features)
    

    def apply_features(self, features, training_data):
        X = np.zeros((len(features), len(training_data)))
        y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        for positive_regions, negative_regions in features:
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) \
                                 - sum([neg.compute_feature(ii) for neg in negative_regions])
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return X, y

    def train_weak(self, X, y, features, weights):
        total_pos, total_neg = 0, 0
        for weight, label in zip(weights, y):
            if label == 1:
                total_pos += weight
            else:
                total_neg += weight
        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0,  0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float("inf"), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1
                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def select_best(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float("inf"), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def train(self, training, pos_num, neg_num):
        weights = np.zeros(len(training))
        training_data = []
        for x in range(len(training)):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)
        features = self.build_features(training_data[0][0].shape)
        X, y = self.apply_features(features, training_data)
        # optimize : pre select features to accelerate
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" %
                  (str(clf), len(accuracy) - sum(accuracy), alpha))

    def classify(self, image):
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(alpha) else 0

    def save(self, file_name):
        with open(file_name+".pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name+".pkl", "rb") as f:
            return pickle.load(f)


# Cascade Haar
class CascadeClassifier():

    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, training):
        pos, neg = [], []
        for ex in training:
            if ex[1] == 1:
                pos.append(ex)
            else:
                neg.append(ex)
        for feature_num in self.layers:
            if len(neg) == 0:
                print("Stopping early. FPR = 0")
            clf = ViolaJones(T=feature_num)
            clf.train(pos+neg, len(pos), len(neg))
            self.clfs.append(clf)
            false_positives = []
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            neg = false_positives

    def classify(self, image):
        for clf in self.clfs:
            if clf.classify(image) == 0:
                return 0
        return 1

    def save(self, file_name):
        with open(file_name+".pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name + ".pkl", "rb") as f:
            return pickle.load(f)