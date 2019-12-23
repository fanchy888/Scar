import math
import random

import numpy as np


class IsolationTreeNode:
    def __init__(self, size=0, attr_idx=None, attr_value=None):
        self.size = size
        self.left = None
        self.right = None
        self.attr_idx = attr_idx
        self.attr_value = attr_value
        self.is_leaf_node = attr_idx is None


class IsolationForest:
    def __init__(self, tree_num=100, sub_size=256):
        self.forest = []
        self.tree_num = tree_num
        self.sub_size = sub_size

    def build_forest(self, x):
        sample_size = len(x)
        self.sub_size = min(sample_size, self.sub_size)
        max_tree_height = math.ceil(math.log(self.sub_size, 2))
        for i in range(self.tree_num):
            indices = np.random.choice(sample_size, self.sub_size, replace=False)
            sub_x = np.array(x)[indices]
            self.forest.append(self._build_tree(sub_x, 0, max_tree_height))

    def _build_tree(self, x, level, max_tree_height):
        m, n = x.shape  # m samples and n attributes
        if level >= max_tree_height or m < 2:
            return IsolationTreeNode(size=m)

        else:
            attr_idx = np.random.randint(0, n)
            low = max(x[:, attr_idx])
            high = min(x[:, attr_idx])
            attr_value = random.uniform(low, high)
            left = x[:, attr_idx] < attr_value
            right = x[:, attr_idx] >= attr_value

            tree = IsolationTreeNode(attr_idx=attr_idx, attr_value=attr_value)
            tree.left = self._build_tree(x[left], level+1, max_tree_height)
            tree.right = self._build_tree(x[right], level + 1, max_tree_height)
            return tree

    def decision(self, x, threshold=0.6):
        if not self.forest:
            self.build_forest(x)

        scores = np.array([self.score(sample) for sample in x])
        return np.where(scores > threshold)[0]

    def score(self, x):
        levels = []
        for tree in self.forest:
            levels.append(self._predict(x, tree, 0))
        avg_l = np.array(levels).mean()
        return 2 ** (-avg_l / self.avg_search_depth(self.sub_size))

    def _predict(self, x, tree, level):
        # single tree decision for one sample
        if tree.is_leaf_node:
            return level + self.avg_search_depth(tree.size)
        else:
            next_node = tree.right if x[tree.attr_idx] >= tree.attr_value else tree.left
            return self._predict(x, next_node, level + 1)

    @staticmethod
    def avg_search_depth(size):
        if size < 2:
            return 0
        else:
            h = math.log(size-1) + 0.5772156649
            # h = functools.reduce(lambda a, b: a + b, [1/x for x in range(1, size)])
            return 2 * h - 2 * (size - 1) / size
