# Understanding SVMs

# The training points that lie on the border between the classes matter for defining the decision boundary.
# These points are called supporting vectors.
# A classification decision is made based on the distances to the support vector, and the importance of
# the support vectors that was learned during training (stored in the dual_coef_ attribute of SVC).

# The distance between data points is measured by the Gaussian kernel:

# krbf(x1, x2) = exp (ɣǁx1 - x2ǁ ^ 2)

# Here, x1 and x2 are data points, ǁ x1 - x2 ǁ denotes Euclidean distance, and ɣ (gamma)
# is a parameter that controls the width of the Gaussian kernel.


from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

# ------------------- UNCOMMENT THIS TO LEARN BASIC ABOUT SVM --------------------------------------#

X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel="rbf", C=10, gamma=0.1).fit(X, y)
# mglearn.plots.plot_2d_separator(svm, X, eps=0.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # plot support vectors
# sv = svm.support_vectors_
# # class labels of support vectors are given by the sign of the dual coefficients
# sv_labels = svm.dual_coef_.ravel() > 0
# mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")


# The decision boundary is shown in black, and the support vectors are larger points with the wide outline.

# The gamma parameter controls the width of the Gaussian kernel. It determines the scale of what it
# means for points to be close together.
# The C parameter is a regularization parameter, similar to that used in the linear models.
# It limits the importance of each point (or more precisely, their dual_coef_)


# -------------------UNCOMMENT THIS TO KNOW HOW PARAMETERS AFFECT SVM --------------------------------------#
# fig, axes = plt.subplots(3, 3, figsize=(15, 10))

# for ax, C in zip(axes, [-1, 0, 3]):
#     for a, gamma in zip(ax, range(-1, 2)):
#         mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

# axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
#  ncol=4, loc=(.9, 1))

                  
# A small gamma means a large radius for the Gaussian kernel, which means that many
# points are considered close by which reflects with smoother boundaries. Thus, it yields a low model complexiety.

# Increasing C, allows the points to have a stronger influence on the model and makes the decision boundary 
# bend to correctly classify them.

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, random_state=0)
svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")

# We can see that the features of the breast_cancer_dataset are of completelt different magnitude. 
# This has devastating effects for the kernel SVM.

# For better understanding:
#       1) Feature Index: the position of a feature in the data. 
#       2) Feature Magnitude: how big or small your feature values are. 

#   If one feature (say "height") has much larger numbers than the others, 
#   it dominates the distance — and SVM will pay too much attention to that feature, ignoring others like "age"
plt.show()
