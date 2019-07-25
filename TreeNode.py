# Decision Tree Tree Node
from DataGenerator import DataSet
import numpy as np


class TreeNode:
    def __init__(self, available_examples_index, available_attributes_index):
        self.available_examples_index = available_examples_index
        self.available_attributes_index = available_attributes_index
        self.label = None
        self.attribute = None
        self.threshold = None
        self.is_leaf = False
        self.children = []
        if len(available_examples_index) == 0:
            return
        self.grow()

    def grow(self):
        if self.is_leaf:
            return
        labels = DataSet.examples[self.available_examples_index][:, -1].tolist()
        # All with the same label, return the label
        unique_labels = np.unique(labels)
        for i in range(len(unique_labels)):
            all_label_x = [x for x in labels if x == unique_labels[i]]
            if len(all_label_x) == 0:
                self.label = unique_labels[i]
                self.is_leaf = True

        if not self.is_leaf:
            # No available attributes, return the most frequent label
            if len(self.available_attributes_index) == 0:
                self.label = max(set(labels), key=labels.count)
                self.is_leaf = True
            # find the best attribute, grow the tree
            else:
                target_attribute_index, target_examples_index, threshold, is_expected = \
                    DataSet.find_best_attribute(self.available_attributes_index, self.available_examples_index)
                if is_expected:
                    self.is_leaf = True
                    self.label = DataSet.most_frequent
                    self.threshold = threshold
                    return
                # else:
                self.attribute = target_attribute_index
                child_attributes = [x for x in self.available_attributes_index if x != target_attribute_index]
                for examples in target_examples_index:
                    node = TreeNode(examples, child_attributes)
                    if len(examples) == 0:
                        node.label = max(set(labels), key=labels.count)
                        node.is_leaf = True
                    self.children.append(node)
                self.threshold = threshold

    def print(self):
        print(self.label, end=" ")
        print(self.attribute, end=" ")
        print(self.is_leaf, end="; ")
