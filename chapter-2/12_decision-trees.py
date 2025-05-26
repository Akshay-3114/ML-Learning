# A decision tree is a machine learning model that makes predictions by asking a 
# series of yes/no questions about the data, and following the answers down a tree-like structure 
# until it reaches a final decision.

# These questions are called test.

# To build a tree, the algorithm searches over all possible tests and finds the one that is
# most informative about the target variable.

# The recursive partitioning of the data is repeated until each region in the partition
# (each leaf in the decision tree) only contains a single target value (a single class or a
# single regression value). A leaf of the tree that contains data points that all share the
# same target value is called pure.

# Building a tree as described here and continuing until all leaves are pure
# leads to models that are very complex and highly overfit to the training data. 


# There are two common strategies to prevent overfitting: stopping the creation of the
# tree early (also called pre-pruning), or building the tree but then removing or collapsing nodes 
# that contain little information (also called post-pruning or just pruning).
# scikit-learn only implements pre-pruning, not post-pruning

# Some of the criteria to pre-prune are:- 
# Limiting the maximum depth of the tree,
# Limiting the maximum number of leaves
# Requiring a minimum number of points in a node to keep splitting it


import mglearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import graphviz

# mglearn.plots.plot_animal_tree()

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))   #1.000
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))   #0.937

# The accuracy on the training set is 100%—because the leaves are pure,
# the tree was grown deep enough that it could perfectly memorize all the labels on the
# training data. While, the test set accuracy is slightly worse than for the linear models(95%).

# Let's try pre-pruning.
# One option is to stop building the tree after a certain depth has been reached.

#Here we set max_depth=4, meaning only four consecutive questions can be asked
tree = DecisionTreeClassifier(max_depth=4, random_state=0) 
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train))) #0.988
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))    #0.951

# We can visualize the tree using the export_graphviz function from the tree module.
# This writes a file in the .dot file format, which is a text file format for storing graphs.

export_graphviz(tree, out_file="C:/Users/AkshayS/Desktop/tree.dot", class_names=["malignant", "benign"],
 feature_names=cancer.feature_names, impurity=False, filled=True)

# We can read this file and visualize it, using the graphviz module

with open("tree.dot") as f:
 dot_graph = f.read()
graphviz.Source(dot_graph)


# One method of inspecting the tree that may be helpful is to find out which path most of the data
# actually takes. The n_samples shown in each node gives the number of samples in that node, 
# while value provides the number of samples per class.