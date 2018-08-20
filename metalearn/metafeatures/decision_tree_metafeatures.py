from collections import Counter

from sklearn.tree import DecisionTreeClassifier

from metalearn.metafeatures.common_operations import profile_distribution

class DecisionTree(object):

    def __init__ (self, X, Y, seed):
        self.level_sizes = []
        self.branch_lengths = []
        self.positions = []
        self.tree = self._get_decision_tree(X, Y, seed)
        self.attributes = [
                            x for x in 
                            Counter(self.tree.feature).values() if x != -2
                           ]
        self.adjacencies = [
                            (left, right) for left, right in 
                            zip(self.tree.children_left, 
                                self.tree.children_right)
                            ]
        
        self._get_general_info()
        self._traverse_tree(0, 0, 0, 0)
        self.width = abs(min(self.positions)) + abs(max(self.positions))

    def _get_decision_tree(self, X, Y, seed):
        estimator = DecisionTreeClassifier(random_state=seed)
        estimator.fit(X, Y)
        return estimator.tree_

    def _get_general_info(self):
        self.n_nodes = self.tree.node_count
        self.tree_height = self.tree.max_depth + 1
        self.level_sizes = [0] * (self.tree_height)
        counts = Counter(self.tree.feature)
        self.leaf_count = counts[-2]

    def _traverse_tree(
        self, curr_node, curr_position, curr_level, curr_branch_length
    ):
        self.level_sizes[curr_level] += 1
        if self.adjacencies[curr_node][0] == self.adjacencies[curr_node][1]:
            self.positions.append(curr_position)
            self.branch_lengths.append(curr_branch_length)
        else:
            self._traverse_tree(
                self.adjacencies[curr_node][0], curr_position-1, curr_level+1, 
                curr_branch_length+1
            )
            self._traverse_tree(
                self.adjacencies[curr_node][1], curr_position+1, curr_level+1, 
                curr_branch_length+1
            )


def get_decision_tree(X, Y, seed):
    return (DecisionTree(X, Y, seed),)


def get_decision_tree_level_sizes(tree):
    level_mean, level_stdev, level_min, _, _, _, level_max = profile_distribution(tree.level_sizes)
    return (level_mean, level_stdev, level_min, level_max)


def get_decision_tree_branch_lengths(tree):
    branch_mean, branch_stdev, branch_min, _, _, _, branch_max = profile_distribution(tree.branch_lengths)
    return (branch_mean, branch_stdev, branch_min, branch_max)


def get_decision_tree_attributes(tree):
    att_mean, att_stdev, att_min, _, _, _, att_max = profile_distribution(tree.attributes)
    return (att_mean, att_stdev, att_min, att_max)


def get_decision_tree_general_info(tree):
    return (tree.n_nodes, tree.leaf_count, tree.tree_height, tree.width)