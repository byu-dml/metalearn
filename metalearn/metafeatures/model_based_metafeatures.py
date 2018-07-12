from collections import Counter

from sklearn.tree import DecisionTreeClassifier

from metalearn.metafeatures.common_operations import profile_distribution

'''
TODO Features to implement
    -tree width
'''


def get_decision_tree(X, Y, random):
    estimator = DecisionTreeClassifier(random_state=random)
    estimator.fit(X, Y)
    return (estimator.tree_,)


def get_decision_tree_level_sizes(tree):
    adjacencies = [(left, right) for left, right in zip(tree.children_left, tree.children_right)]
    curr_level = [0]
    level_sizes = [1]
    while len(curr_level) > 0:
        next_level = []
        for node in curr_level:
            if adjacencies[node][0] != adjacencies[node][1]:
                next_level.append(adjacencies[node][0])
                next_level.append(adjacencies[node][1])
        level_sizes.append(len(next_level))
        curr_level = next_level
    level_mean, level_stdev, level_min, _, _, _, level_max = profile_distribution(level_sizes)
    return (level_mean, level_stdev, level_min, level_max)


def get_decision_tree_branch_lengths(tree):
    adjacencies = [(left, right) for left, right in zip(tree.children_left, tree.children_right)]
    leaves = [node for node, children in enumerate(adjacencies) if children[0] == children[1]]
    branch_lengths = []
    for leaf in leaves:
        curr_node = leaf
        length = 0
        while curr_node != 0:
            parent_node = [parent for parent, children in enumerate(adjacencies) if curr_node in children][0]
            length += 1
            curr_node = parent_node
        branch_lengths.append(length)
    branch_mean, branch_stdev, branch_min, _, _, _, branch_max = profile_distribution(branch_lengths)
    return (branch_mean, branch_stdev, branch_min, branch_max)


def get_decision_tree_attributes(tree):
    counts = Counter(tree.feature)
    att_counts = [x for x in counts.values() if x != -2]
    att_mean, att_stdev, att_min, _, _, _, att_max = profile_distribution(att_counts)
    return (att_mean, att_stdev, att_min, att_max)


def get_decision_tree_general_info(tree):
    n_nodes = tree.node_count
    tree_height = tree.max_depth + 1
    counts = Counter(tree.feature)
    leaf_count = counts[-2]
    return (n_nodes, tree_height, leaf_count)
