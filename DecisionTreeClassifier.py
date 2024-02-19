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
        self.val = value


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
    

    def get_best_split(self, data, num_samples, num_features):
        '''function to find best split'''

        best_split = {}
        max_gained_info = -float('inf')

        for index in range(num_features):
            values = data[:, index]
            poss_threshold = np.unique(values)

            for threshold in poss_threshold:
                data_left, data_right = self.split(data, index, threshold)

                if len(data_left) > 0 and len(data_right) > 0:
                    y, left_y, right_y = data[:, -1], data_left[:, -1], data_right[:, -1]

                    curr_gained_info = self.gain_info(y, left_y, right_y, 'gini')

                    if curr_gained_info > max_gained_info:
                        best_split["index"] = index
                        best_split["threshold"] = threshold
                        best_split["data_left"] = data_left
                        best_split["data_right"] = data_right
                        best_split["gained_info"] = curr_gained_info
                        max_info_gain = curr_gained_info
        
        return best_split
    

    def split(self, data, index, treshold):
        '''function that actually splits data'''

        data_left = np.array([r for r in data if r[index] <= treshold])
        data_right = np.array([r for r in data if r[index] > treshold])

        return data_left, data_right
    

    def gain_info(self, y, left_y, right_y, mode='entropy'):
        '''function to compute gained info'''

        weight_l = len(left_y) / len(y)
        weight_r = len(right_y) / len(y)

        if mode == 'gini':
            gain = self.gini_index(y) - (weight_l * self.gini_index(left_y) + weight_r * self.gini_index(right_y))
            
        else:
            gain = self.entropy(y) - (weight_l * self.entropy(left_y) + weight_r * self.entropy(right_y))
        
        return gain
    

    def gini_index(self, y):
        '''function to calculate gini index'''

        labels = np.unique(y)
        gini = 0

        for name in labels:
            gini += (len(y[y == name]) / len(y)) ** 2
        
        return 1 - gini


    def entropy(self, y):
        '''function to calculate entropy'''

        labels = np.unique(y)
        entropy = 0

        for name in labels:
            p = len(y[y == name]) / len(y)
            entropy += -p * np.log2(p)
        
        return entropy
    

    def calc_leaf(self, y):
        '''leaf value calculator'''
        y = list(y)

        return max(y, key=y.count)


    def print_tree(self, tree=None, indent=' '):
        '''function to visualize tree'''

        if not tree:
            tree = self.root
        
        if tree.val != None:
            print(tree.val)

        else:
            print("x_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


    def fit(self, x, y):
        '''function to train a tree'''

        data = np.concatenate((x, y), axis=1)
        self.root = self.build_tree(data)


    def predict(self, x):
        '''calls multiple functions to make predictions'''

        pred = [self.make_pred(xi, self.root) for xi in x]

        return pred
    
    
    def make_pred(self, x, root):
        '''function to make prediction'''

        if root.val != None:
            return root.val
        
        feat_val = x[root.feature_index]

        if feat_val <= root.threshold:
            return self.make_pred(x, root.left)
        
        else:
            return self.make_pred(x, root.right)
        