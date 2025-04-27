##################################      Linear models for classification    ###########################################

#         ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b > 0

# We threshold the predicted value at zero.
# If the function is smaller than zero, we predict the class –1;
# If it is larger than zero, we predict the class +1.
# This prediction rule is common to all linear models for classification.

# For linear models for classification,
# the decision boundary is a linear function of the input. In other words, a (binary) linear classifier is a 
# classifier that separates two classes using a line, a plane, or a hyperplane.


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn.datasets
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
 clf = model.fit(X, y)
 mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
 ax=ax, alpha=.7)
 mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
 ax.set_title("{}".format(clf.__class__.__name__))
 ax.set_xlabel("Feature 0")
 ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()
