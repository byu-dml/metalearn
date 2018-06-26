import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from metalearn.metafeatures.common_operations import profile_distribution

'''
FEATURES TO IMPLEMENT:
    -tree width
    -max/min/mean/stdev of nodes in a level
    -max/min/mean/stdev of branch lengths
'''

def get_model_info(X,Y):

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    estimator = DecisionTreeClassifier(random_state=0)
    estimator.fit(X_train, y_train)

    n_nodes = estimator.tree_.node_count
    tree_height = estimator.tree_.max_depth + 1
    leaf_count = 0
    for feature in estimator.tree_.feature:
        if feature == -2:
            leaf_count += 1
        else:
            pass
    counts = Counter(estimator.tree_.feature)
    leaf_count = counts[-2]
    del counts[-2]
    counts = list(counts.values())

    att_mean, att_stdev, att_min,_,_,_,att_max = profile_distribution(counts)


    # print(f'Leaves: {leaf_count}')
    # print(f'Nodes: {n_nodes}')
    # print(f'Tree Height: {tree_height}')
    # print(f'Attribute Occurence: {att_min}(min), {att_max}(max), {att_mean}(mean), {att_stdev}(stdev) ')

    print(estimator.tree_.value)



# ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', 
# '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', 
# '__pyx_vtable__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', 
# '__subclasshook__', 'apply', 'capacity', 'children_left', 'children_right', 'compute_feature_importances', 
# 'decision_path', 'feature', 'impurity', 'max_depth', 'max_n_classes', 'n_classes', 'n_features', 'n_node_samples', 
# 'n_outputs', 'node_count', 'predict', 'threshold', 'value', 'weighted_n_node_samples']



# # The decision estimator has an attribute called tree_  which stores the entire
# # tree structure and allows access to low level attributes. The binary tree
# # tree_ is represented as a number of parallel arrays. The i-th element of each
# # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# # Some of the arrays only apply to either leaves or split nodes, resp. In this
# # case the values of nodes of the other type are arbitrary!
# #
# # Among those arrays, we have:
# #   - left_child, id of the left child of the node
# #   - right_child, id of the right child of the node
# #   - feature, feature used for splitting the node
# #   - threshold, threshold value at the node
# #

# # Using those arrays, we can parse the tree structure:

# n_nodes = estimator.tree_.node_count
# children_left = estimator.tree_.children_left
# children_right = estimator.tree_.children_right
# feature = estimator.tree_.feature
# threshold = estimator.tree_.threshold


# # The tree structure can be traversed to compute various properties such
# # as the depth of each node and whether or not it is a leaf.
# node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# stack = [(0, -1)]  # seed is the root node id and its parent depth
# while len(stack) > 0:
#     node_id, parent_depth = stack.pop()
#     node_depth[node_id] = parent_depth + 1

#     # If we have a test node
#     if (children_left[node_id] != children_right[node_id]):
#         stack.append((children_left[node_id], parent_depth + 1))
#         stack.append((children_right[node_id], parent_depth + 1))
#     else:
#         is_leaves[node_id] = True

# print("The binary tree structure has %s nodes and has "
#       "the following tree structure:"
#       % n_nodes)
# for i in range(n_nodes):
#     if is_leaves[i]:
#         print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
#     else:
#         print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
#               "node %s."
#               % (node_depth[i] * "\t",
#                  i,
#                  children_left[i],
#                  feature[i],
#                  threshold[i],
#                  children_right[i],
#                  ))
# print()

# # First let's retrieve the decision path of each sample. The decision_path
# # method allows to retrieve the node indicator functions. A non zero element of
# # indicator matrix at the position (i, j) indicates that the sample i goes
# # through the node j.

# node_indicator = estimator.decision_path(X_test)

# # Similarly, we can also have the leaves ids reached by each sample.

# leave_id = estimator.apply(X_test)

# # Now, it's possible to get the tests that were used to predict a sample or
# # a group of samples. First, let's make it for the sample.

# sample_id = 0
# node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                     node_indicator.indptr[sample_id + 1]]

# print('Rules used to predict sample %s: ' % sample_id)
# for node_id in node_index:
#     if leave_id[sample_id] != node_id:
#         continue

#     if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
#         threshold_sign = "<="
#     else:
#         threshold_sign = ">"

#     print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
#           % (node_id,
#              sample_id,
#              feature[node_id],
#              X_test[sample_id, feature[node_id]],
#              threshold_sign,
#              threshold[node_id]))

# # For a group of samples, we have the following common node.
# sample_ids = [0, 1]
# common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
#                 len(sample_ids))

# common_node_id = np.arange(n_nodes)[common_nodes]

# print("\nThe following samples %s share the node %s in the tree"
#       % (sample_ids, common_node_id))
# print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))