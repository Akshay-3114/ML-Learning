# Kernelized support vector machines
# (often just referred to as SVMs) are an extension that allows for more complex models that 
# are not defined simply by hyperplanes in the input space

from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC   
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = make_blobs(centers=4, random_state=8)
y = y % 2

# 1) Without any model. Only graphical representation of the points.

# 2) using the linear SVM with 2 Dimensions(features)
linear_svm = LinearSVC().fit(X, y) 

# A linear model for classification can only separate points using a line, and will not be
# able to do a very good job on this dataset

# ---------------------UNCOMMENT THIS FOR 1 AND 2 ------------------------------------#

# mglearn.plots.plot_2d_separator(linear_svm, X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")


# 3) using 3 Dimensions, [X1, X2, X1^2]
X_new = np.hstack([X, X[:, 1:] ** 2])

# 4) Using Linear SVM with 3D
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

figure = plt.figure()
# visualize in 3D
ax = Axes3D(figure, elev=-152, azim=-26)
# plot first all the points with y == 0, then all with y == 1


# 4) 
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)


mask = y == 0

ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")


# 6) As a function of the original features, the linear SVM model is not actually linear any‐
# more. It is not a line, but more of an ellipse
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

plt.show()

#Thus, adding nonlinear features to the representation of our data can make linear models much more powerful. 
# Often we don’t know which features to add, and adding many features might make computation very expensive.

# The kernel trick works by directly computing the distance (more precisely, the scalar products) of the 
# data points for the expanded feature representation, without ever actually computing the expansion.

# Two ways to map your data into a higher-dimensional space that are commonly used with support vector machines:

# 1) The polynomial kernel, which computes all possible polynomials up to a certain degree of the 
#        original features (like feature1 ** 2 * feature2 ** 5)

# 2) The radial basis function (RBF) kernel, also known as the Gaussian kernel. It considers all possible 
#       polynomials of all degrees, but the importance of the features decreases 
#       for higher degrees(following the Taylor expansion of the exponential map).