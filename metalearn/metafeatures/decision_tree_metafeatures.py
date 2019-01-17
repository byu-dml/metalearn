from collections import Counter

from sklearn.tree import DecisionTreeClassifier

from metalearn.metafeatures.common_operations import profile_distribution

class DecisionTree:

    def __init__(self, X, Y, seed):
        self.tree = self._get_decision_tree(X, Y, seed)
        self.n_nodes = self.tree.node_count
        self.tree_height = self.tree.max_depth + 1
        self.leaf_count = Counter(self.tree.feature)[-2]  # -2 indicates that a node has no children, ie is a leaf node

    def _get_decision_tree(self, X, Y, seed):
        estimator = DecisionTreeClassifier(random_state=seed)
        estimator.fit(X, Y)
        return estimator.tree_

    def get_general_info(self):
        return self.n_nodes, self.leaf_count, self.tree_height

    def get_attributes(self):
        return [x for x in Counter(self.tree.feature).values() if x != -2]

class TraversedDecisionTree:

    def __init__(self, tree):
        self.tree = tree
        self.branch_lengths = []
        self.positions = []
        self.adjacencies = [
                            (left, right) for left, right in
                            zip(self.tree.tree.children_left,
                                self.tree.tree.children_right)
                            ]
        self.level_sizes = [0] * self.tree.tree_height
        self._traverse_tree(0, 0, 0, 0)

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

    def get_width(self):
        return abs(min(self.positions)) + abs(max(self.positions))


def get_decision_tree(X, Y, seed):
    return (DecisionTree(X, Y, seed),)

def traverse_tree(tree):
    return (TraversedDecisionTree(tree),)

def get_decision_tree_level_sizes(tree):
    return profile_distribution(tree.level_sizes)

def get_decision_tree_branch_lengths(tree):
    return profile_distribution(tree.branch_lengths)

def get_decision_tree_attributes(tree):
    return profile_distribution(tree.get_attributes())

def get_decision_tree_general_info(tree):
    return tree.get_general_info()

def get_decision_tree_width(tree):
    return (tree.get_width(),)
