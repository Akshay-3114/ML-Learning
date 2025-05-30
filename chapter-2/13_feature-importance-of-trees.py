from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mglearn

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=0) 
tree.fit(X_train, y_train)

# There are some useful properties that we can derive to summarize
# the workings of the tree. Feature importance, which rates how important each 
# feature is for the decision a tree makes.

# It is a number between 0 and 1 for each feature, where 0
# means “not used at all” and 1 means “perfectly predicts the target.” The feature
# importances always sum to 1.

print("Feature importances:\n{}".format(tree.feature_importances_))

def plot_feature_importances_cancer(model):
 n_features = cancer.data.shape[1]
 plt.barh(range(n_features), model.feature_importances_, align='center')
 plt.yticks(np.arange(n_features), cancer.feature_names)
 plt.xlabel("Feature importance")
 plt.ylabel("Feature")
 plt.show()
plot_feature_importances_cancer(tree)

tree = mglearn.plots.plot_tree_not_monotone()
display(tree)


#  Tree models don't jell well with regression tasks 
#   as they are incapable of generating new responses.

#  The tree model makes perfect predictions on the training data with no restriction
# However, once we leave the data range for which
# the model has data, the model simply keeps predicting the last known point. The tree
# has no ability to generate “new” responses, outside of what was seen in the training
# data. This shortcoming applies to all models based on trees.


# Note:-

# It is actually possible to make very good forecasts with tree-based models (for example, when trying to predict
# whether a price will go up or down).


# Parameter :- Used to control the model complexity of tree. Specifically pre-pruning.
# max_depth, max_leaf_nodes, or min_samples_leaf—are sufficient to prevent overfitting.

# Decision trees have two advantages over many of the algorithms
#     1) The resulting model can easily be visualized and understood by nonexperts (at
#        least for smaller trees).
#     2) The algorithms are completely invariant to scaling of the data.
#        As each feature is processed separately, and the possible splits of the data 
#        don’t depend on scaling, no preprocessing like normalization or 
#        standardization of features is needed for decision tree algorithms.