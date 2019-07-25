# C4.5 decision tree classifier
from TreeNode import TreeNode
import collections
from DataGenerator import DataSet
import numpy as np


class Feature:
    def __init__(self, measure, op, attribute_index, label=None):
        self.measure = measure
        self.op = op
        self.label = label
        self.attribute_index = attribute_index

    def judge(self, v):
        # do not judge a label
        assert self.label is None
        d = {
            '==': v == self.measure,
            '<=': v <= self.measure,
            '>': v > self.measure
        }
        return d[self.op]

    def print(self):
        if self.label is None:
            print(str(self.attribute_index) + ': ' + self.op + str(self.measure), end=' ')
        else:
            print('label: ' + str(self.label))


class MyDecisionTree:
    def __init__(self):
        self.root = None

    def train(self, examples_index, attributes_index):
        self.root = TreeNode(examples_index, attributes_index)

    def predict(self, example):
        current = self.root
        if not DataSet.is_category:
            while not current.is_leaf:
                if example[current.attribute] <= current.threshold:
                    current = current.children[0]
                else:
                    current = current.children[1]
        else:
            while not current.is_leaf:
                labels = DataSet.examples[:, current.attribute].tolist()
                values = np.unique(labels)
                for i in range(len(values)):
                    # print(example[current.attribute])
                    # print(current.attribute)
                    if example[current.attribute] == values[i]:
                        current = current.children[i]
                        break
        return current.label

    def print(self):
        to_print = collections.deque()
        to_print.append(self.root)

        while to_print:
            this_line = []
            while to_print:
                node = to_print.popleft()
                this_line.append(node)
                node.print()
            print()
            for n in this_line:
                if not n.is_leaf:
                    for c in n.children:
                        to_print.append(c)

    def to_rules(self, rule, node):
        if node.is_leaf:
            f = Feature(None, None, None, node.label)
            r = rule.copy()
            r.append(f)
            DataSet.rules.append(r)
        else:
            if not DataSet.is_category:
                f1 = Feature(node.threshold, '<=', node.attribute)
                f2 = Feature(node.threshold, '>', node.attribute)
                r1 = rule.copy()
                r1.append(f1)
                r2 = rule.copy()
                r2.append(f2)
                self.to_rules(r1, node.children[0])
                self.to_rules(r2, node.children[1])
            else:
                labels = DataSet.examples[:, node.attribute].tolist()
                values = np.unique(labels)
                for i in range(len(values)):
                    f = Feature(values[i], '==', node.attribute)
                    r = rule.copy()
                    r.append(f)
                    self.to_rules(r, node.children[i])
