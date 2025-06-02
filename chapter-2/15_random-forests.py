################################# Random Forests #####################

# Random forests are used to overcome the drawback of overfitting in regular decision trees.

# A random forest is essentially a collection of decision trees, 
# where each tree is slightly different from the others. 

# The idea behind random forests is that each tree might do a relatively
# good job of predicting, but will likely overfit on part of the data. If we build many
# trees, all of which work well and overfit in different ways, we can reduce the amount
# of overfitting by averaging their results.
# Random forests get their name from injecting randomness into the tree building 
# to ensure each tree is different. There are two ways to introduce this randomness

    # 1)By selecting the data points used to build a tree 
    # 2)By selecting the features in each split test


# Building random forests. We need to

    # 1) Decide on the number of trees to build(n_estimators)
    # 2) Bootstrap sample of our data. That is, from our n_samples data points, 
    #    we repeatedly draw an example randomly with replacement (meaning the 
    #    same sample can be picked multiple times), n_samples times.

        #  Instead of looking for the best test for each node, in each node the algorithm randomly selects 
        #  a subset of the features, and it looks for the best possible test involving one of these features.
    # 3) The number of features that are selected is controlled by the max_features parameter. 

    # The bootstrap sampling leads to each decision tree in the random forest being built on a slightly 
    # different dataset. Because of the selection of features in each node, each split in each tree operates
    # on a different subset of features. Together, these two mechanisms ensure that all the trees in 
    # the random forest are different.

# A high max_features means that the trees in the random forest will be quite similar,and they will be
# able to fit the data easily, using the most distinctive features. A low max_features
# means that the trees in the random forest will be quite different, and that each tree
# might need to be very deep in order to fit the data well.

# To make a prediction using the random forest, the algorithm first makes a prediction
# for every tree in the forest. For regression, we can average these results to get our final
# prediction. For classification, a “soft voting” strategy is used. This means each algorithm makes 
# a “soft” prediction, providing a probability for each possible output label. The probabilities 
# predicted by all the trees are averaged, and the class with the highest probability is predicted.

# Summary
# Step	|    What Happens
# 1.    |  Bootstrapping	Randomly sample rows for each tree (with replacement)
# 2.    |  Random Features	At each split, consider a random subset of features
# 3.    |  Grow Tree	Build a deep tree using the sample and subset of features
# 4.    |  Predict	Combine all tree outputs using voting (classification) or averaging (regression)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = make_moons(n_samples=100,noise=0.25, random_state=51)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2) # we are building a forest of 5 trees
forest.fit(X_train, y_train)

# The graphs display the differences in the trees and their decision boundaries along with the aggregated 
# boundary of the random forest.

# ----- UNCOMMENT AND RUN THE BELOW LINES TO SEE THE GRAPH -----------------

# fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
#  ax.set_title("Tree {}".format(i))
#  mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
# mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
#  alpha=.4)
# axes[-1, -1].set_title("Random Forest")
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.show()

# The decision boundaries learned by the five trees are quite different. Each of them makes some mistakes, 
# as some of the training points that are plotted here were not actually included in the training sets of 
# the trees, due to the bootstrap sampling.

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


print("Feature importances:\n{}".format(forest.feature_importances_))

def plot_feature_importances_cancer(model):
 n_features = cancer.data.shape[1]
 plt.barh(range(n_features), model.feature_importances_, align='center')
 plt.yticks(np.arange(n_features), cancer.feature_names)
 plt.xlabel("Feature importance")
 plt.ylabel("Feature")
 plt.show()
plot_feature_importances_cancer(forest)





#--------------------------------Strengths, weaknesses, and parameters--------------------------------------#

# ---> They are very powerful, often work well without heavy tuning of the parameters, and don’t require scaling 
        # of the data.
# ---> While building random forests on large datasets might be somewhat time consuming, it can be parallelized 
#       across multiple CPU cores within a computer easily. You can use the n_jobs parameter to adjust the number 
#       of cores to use.
# ---> To have reproducible results, it is important to fix the random_state.

# ---> Random forests don’t tend to perform well on very high dimensional, sparse data, such as text data. 
#       For this kind of data, linear models might be more appropriate. This is because, it requires more 
#       memory and are slower to train and to predict than linear models.


# ---> For n_estimators, larger is always better. Averaging more trees will yield a more robust 
#       ensemble by reducing overfitting. A common rule of thumb is to build “as many as you have time/memory for.”
# ---> max_features determines how random each tree is, and a smaller max_features reduces overfitting.
#       It’s a good rule of thumb to use the default values: 
#       max_features=sqrt(n_features) for classification and max_features=log2(n_features) for regression.