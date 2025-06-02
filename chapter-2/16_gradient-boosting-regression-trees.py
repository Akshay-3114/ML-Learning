###############################  Gradient Boosting Regression trees #######################################

# Gradient boosting works by building trees in a serial manner, where each tree tries to 
# correct the mistakes of the previous one. By default, there's no randomization, instead strong pre-pruning 
# is used.

# Gradient boosted trees often use very shallow trees (weak learners), of depth one to five,
# which makes the model smaller in terms of memory and makes predictions faster.

# The learning_rate, which controls how strongly each tree tries to correct the mistakes of the previous trees. 
# A higher learning rate means each tree can make stronger corrections, allowing for more complex models.
# n_estimators also increases the complexiety, as the model has more chances to correct mistakes on the training set.


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer()
X_train, X_test , y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

# 1) Without pre-pruning
print("Without pre-pruning")
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("Accuracy on the training set:{:.3f}".format(gbrt.score(X_train, y_train))) #1.00, overfitting
print("Accuracy on the test set:{:.3f}".format(gbrt.score(X_test, y_test))) # 0.958

# 2) Pre-pruning with max_depth = 1
print("\nPre-pruning with max_depth")
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("Accuracy on the training set:{:.3f}".format(gbrt.score(X_train, y_train))) #0.991
print("Accuracy on the test set:{:.3f}".format(gbrt.score(X_test, y_test))) # 0.965

def plot_feature_importances_cancer(model):
 n_features = cancer.data.shape[1]
 plt.barh(range(n_features), model.feature_importances_, align='center')
 plt.yticks(np.arange(n_features), cancer.feature_names)
 plt.xlabel("Feature importance")
 plt.ylabel("Feature")
 plt.show()
plot_feature_importances_cancer(gbrt)

# 3) Pre-pruning with learning_rate = 0.01
print("\nPre-pruning with learning rate")
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print("Accuracy on the training set:{:.3f}".format(gbrt.score(X_train, y_train))) #0.993
print("Accuracy on the test set:{:.3f}".format(gbrt.score(X_test, y_test))) # 0.958

# Both GBRT and Random forest classifiers perform well on similar data. So a common approach is try random forest.
# If random forests work well but prediction time is at a premium, or it is important to squeeze out 
# the last percentage of accuracy from the machine learning model, moving to gradient boosting often helps.

# xgboost package can be used for a large scaled problem.

#--------------------------------Strengths, weaknesses, and parameters--------------------------------------#

# ---> Their main drawback is that they require careful tuning of the parameters and may take a long time to
#       train. Similar to other tree models, it works well without scaling and on mixture of binary or continuous feature.
#       but also often does not work well on high-dimensional sparse data. 

# ---> A lower learning_rate means that more trees are needed to build a model of similar complexity.
#         a higher n_estimators  in gradient boosting leads to a more complex model, which may lead to overfitting.

        # A common practice is to fit n_estimators depending on the time and memory budget, 
        # and then search over different learning_rates.
# ---> Usually max_depth is set very low for gradient boosted models, often not deeper than five splits.