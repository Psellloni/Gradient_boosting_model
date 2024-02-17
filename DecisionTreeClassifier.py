import numpy as np

class TreeNode:
    def __init__(self, feature_index=None, treshold=None, left=None, right=None, gained_info=None, value=None):
        '''constructor'''

        # decision node
        self.feature_index = feature_index
        self.treshold = treshold
        self.left = left
        self.right = right
        self.gained_info = gained_info

        # leaf node
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_sample_split=2, max_depth=2):
        '''constructor'''

        self.root = None

        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
    
    def build_tree(self, data, current_depth=0):
        '''recursive function to build the tree'''

        x, y = data[:,:-1], data[:,-1]
        num_samples, num_features = np.shape(x)

        if num_samples >= self.min_sample_split and current_depth <= self.max_depth:
            best_split = self.get_best_split(data, num_samples, num_features)

            if best_split['gained_info'] > 0:
                left_subtree = self.build_tree(best_split['left_data'], current_depth + 1)
                right_subtree = self.build_tree(best_split['right_data'], current_depth + 1)

                return TreeNode(best_split['feature_index'], best_split['treshold'], left_subtree, 
                                right_subtree, best_split['gained_info'])

        leaf_value = self.calc_leaf(y)

        return TreeNode(value=leaf_value)

