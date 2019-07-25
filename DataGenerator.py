# generate examples and attributes
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import entropy
import numpy as np
import math
import random


class DataGenerator:
    def __init__(self, file_path, is_category, noise, noise_type):
        self.file_path = file_path
        self.is_category = is_category
        self.noise = noise
        self.noise_type = noise_type
        self.generate()

    # Return: attributes (1, )
    # Return: examples (m, n) The last column is label
    def generate(self):
        if self.is_category:
            df = pd.read_csv(self.file_path, header=None, na_values='b')
            df = df.replace('x', 1)
            df = df.replace('o', 2)
            # df = df.replace('b', 3)
            # print(df.values)
            df = df.replace('positive', 1)
            df = df.replace('negative', 0)

            # impute blank values with most_frequent values
            imp_mean = SimpleImputer(strategy='most_frequent')
            imp_mean.fit(df.iloc[:, 0:df.shape[1] - 1])
            df.iloc[:, 0:df.shape[1] - 1] = imp_mean.transform(df.iloc[:, 0:df.shape[1] - 1])
            df = df.astype('int8')

            # TODO: make some noise
            if self.noise_type == 1:
                noise_sample = random.sample(range(df.shape[0]), int(self.noise * df.shape[0] / 100))

                for i in noise_sample:
                    df.iat[i, -1] = (df.iat[i, -1] + 1) % 2
                DataSet(df.values, df.iloc[:, 0:df.shape[1] - 1].columns.values, True)
            else:
                values = df.values.tolist().copy()
                noise_sample = random.sample(range(df.shape[0]), int(self.noise * df.shape[0] / 100))
                for i in noise_sample:
                    temp = values[i].copy()
                    temp[-1] = temp[-1] % 2 + 1
                    values.append(temp)
                DataSet(np.asarray(values), df.iloc[:, 0:df.shape[1] - 1].columns.values, True)

        else:
            df = pd.read_csv(self.file_path, na_values="None", header=None).iloc[:, 1:]
            imp_mean = SimpleImputer(strategy='mean')
            imp_mean.fit(df.iloc[:, 0:df.shape[1] - 1])
            df.iloc[:, 0:df.shape[1] - 1] = imp_mean.transform(df.iloc[:, 0:df.shape[1] - 1])

            # TODO: make some noise
            if self.noise_type == 1:
                noise_sample = random.sample(range(df.shape[0]), int(self.noise * df.shape[0] / 100))
                # for i in range(len(df.values.tolist())):
                #     if df.iat[i, -1] == 7:
                #         df.iat[i, -1] = 4
                for i in noise_sample:
                    df.iat[i, -1] = (df.iat[i, -1] + 2) % 6 + 1

                DataSet(df.values, df.iloc[:, 0:df.shape[1] - 1].columns.values, False)
            else:
                values = df.values.tolist().copy()
                # for i in range(len(values)):
                #     if values[i][-1] == 7:
                #         values[i][-1] = 4
                noise_sample = random.sample(range(df.shape[0]), int(self.noise * df.shape[0] / 100))
                for i in noise_sample:
                    temp = values[i].copy()
                    temp[-1] = (temp[-1] + 2) % 6 + 1
                    values.append(temp)
                DataSet(np.asarray(values), df.iloc[:, 0:df.shape[1] - 1].columns.values, False)


class DataSet:
    examples = None
    attributes = None
    most_frequent = None
    is_category = False
    rules = []

    def __init__(self, examples, attributes, is_category):
        DataSet.is_category = is_category
        DataSet.examples = examples
        DataSet.attributes = attributes
        labels = DataSet.examples[:, -1].tolist()
        if DataSet.is_category:
            DataSet.most_frequent = max(set(labels), key=labels.count)

    @staticmethod
    def find_best_attribute(attributes_index, examples_index):
        if DataSet.is_category:
            # categorical values
            max_information_gain = -np.inf
            max_intrinsic_value = -np.inf
            target_attribute_index = 0
            target_examples_list = []
            gains = []
            values = []
            for i in attributes_index:
                subs = DataSet.split_with_attribute(examples_index, i)
                current_information_gain, current_intrinsic_value = DataSet.information_gain(subs)
                gains.append(current_information_gain)
                values.append(current_intrinsic_value)

            average_gain = np.average(gains)
            # if average_gain == 0:
            #     return target_attribute_index, target_examples_list, 1, True

            for i in range(len(gains)):
                if gains[i] >= average_gain:
                    if gains[i] / values[i] >= max_intrinsic_value:
                        max_intrinsic_value = gains[i] / values[i]
                        target_attribute_index = attributes_index[i]
                        subs = DataSet.split_with_attribute(examples_index, attributes_index[i])
                        target_examples_list = subs

            # print(len(target_examples_list))
            return target_attribute_index, target_examples_list, 1, False
        else:
            # continuous values
            max_information_gain = -np.inf
            max_intrinsic_value = -np.inf
            target_threshold = None
            target_attribute_index = 0
            target_examples_list = []
            gains = []
            values = []
            threshold_list = []
            attributes = []
            for i in attributes_index:
                current_column = DataSet.examples[:, i]
                sorted_column = np.sort(current_column)
                for j in range(1, len(sorted_column)):
                    if sorted_column[j - 1] != sorted_column[j]:
                        threshold = (sorted_column[j - 1] + sorted_column[j]) / 2
                        subs = DataSet.split_with_attribute(examples_index, i, threshold)
                        current_information_gain, current_intrinsic_value = DataSet.information_gain(subs)
                        gains.append(current_information_gain)
                        values.append(current_intrinsic_value)
                        threshold_list.append(threshold)
                        attributes.append(i)

            average_gain = np.average(gains)
            # if average_gain < 0.001:
            #     return target_attribute_index, target_examples_list, 1, True
            for i in range(len(gains)):
                if gains[i] >= average_gain:
                    if gains[i] / values[i] >= max_intrinsic_value:
                        max_intrinsic_value = gains[i] / values[i]
                        target_attribute_index = attributes[i]
                        target_threshold = threshold_list[i]
                        subs = DataSet.split_with_attribute(examples_index, attributes[i], threshold_list[i])
                        target_examples_list = subs

            return target_attribute_index, target_examples_list, target_threshold, False

    # split with attribute
    # examples: to be splited
    # attribute: split based on this attribute
    # Return: pos, neg
    @staticmethod
    def split_with_attribute(examples_index, attribute_index, threshold=1):
        if not DataSet.is_category:
            pos, neg = [], []
            # features = examples_index[0:len(examples_index) - 2]
            for i in examples_index:
                if DataSet.examples[i][attribute_index] <= threshold:
                    pos.append(i)
                else:
                    neg.append(i)
            return [pos, neg]
        else:
            label_list = DataSet.examples[:, attribute_index]
            value, count = np.unique(label_list, return_counts=True)
            res = []
            for i in range(len(value)):
                res.append([])
            for i in examples_index:
                for j in range(len(value)):
                    if DataSet.examples[i][attribute_index] == value[j]:
                        res[j].append(i)
            return res

    # @staticmethod
    # def information_gain(left_example_index, right_example_index):
    #     left_labels = DataSet.examples[left_example_index][:, -1].tolist()
    #     right_labels = DataSet.examples[right_example_index][:, -1].tolist()
    #
    #     left_value, left_count = np.unique(left_labels, return_counts=True)
    #     right_value, right_count = np.unique(right_labels, return_counts=True)
    #     total_value, total_count = np.unique(left_labels + right_labels, return_counts=True)
    #
    #     left_entropy = entropy(left_count, base=2)
    #     right_entropy = entropy(right_count, base=2)
    #     total_entropy = entropy(total_count, base=2)
    #
    #     information_gain = total_entropy \
    #                        - (len(left_labels) / (len(left_labels) + len(right_labels))) * left_entropy \
    #                        - (len(right_labels) / (len(left_labels) + len(right_labels))) * right_entropy
    #     return information_gain

    @staticmethod
    def information_gain(example_index_list):
        labels_list = []
        for example_index in example_index_list:
            labels = DataSet.examples[example_index][:, -1].tolist()
            labels_list.append(labels)

        total_label = []
        entropy_list = []
        for labels in labels_list:
            total_label += labels
            value, count = np.unique(labels, return_counts=True)
            entropy_list.append(entropy(count, base=2))

        total_value, total_count = np.unique(total_label, return_counts=True)
        total_entropy = entropy(total_count, base=2)

        information_gain = total_entropy
        # if total_entropy == 0:
        #     return 0, 1
        intrinsic_value = 0
        for i in range(len(labels_list)):
            p = len(labels_list[i]) / len(total_label)
            information_gain = information_gain - p * entropy_list[i]
            if p == 0:
                intrinsic_value = np.inf
                continue
            intrinsic_value = intrinsic_value - p * math.log(p, 2)
        return information_gain, intrinsic_value

    @staticmethod
    def print_rules(rules, index_list):
        for i in index_list:
            assert i < len(DataSet.rules)
            rule = DataSet.rules[i]
            for j in range(len(rule)):
                if rule[j] is None:
                    print("None", end=' ')
                else:
                    feature = rule[j]
                    feature.print()

