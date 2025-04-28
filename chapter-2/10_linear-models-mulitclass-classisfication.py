# A common technique to extend a binary classification algorithm to a multiclass classification
# algorithm is the one-vs-rest approach. 
# In the one-vs.-rest approach, a binary model is learned for each class that tries to separate 
# that class from all of the other classes, resulting in as many binary models as there are classes.

# The prediction is made on which classifier results with the highest confidence score.


from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import mglearn
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(random_state=42)

# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# Example to show the inital dataset
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(["Class 0", "Class 1", "Class 2"])

linear_svm = LinearSVC().fit(X, y)
print("Coefficient shape: ", linear_svm.coef_) 
# Here there are two features. 
#       1) If F0 is neg and F1 is pos, then it belongs to class 0
#       2) If F0 is pos and F1 is neg, then it belongs to class 1
#       1) If F0 is neg and F1 is neg, then it belongs to class 2

print("Intercept shape: ", linear_svm.intercept_)

line = np.linspace(-15, 15)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7) #color fills the graph to understand the regions for each class

for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
 'Line class 2'], loc=(1.01, 0.3))

plt.show()



#Conclusion notes

# 1)The main parameter of linear models is the regularization parameter, called alpha in
#   the regression models and C in LinearSVC and LogisticRegression. Large values for
#   alpha or small values for C mean simple models

# 2)The other decision you have to make is whether you want to
#   use L1 regularization or L2 regularization. If you assume that only a few of your fea‐
#   tures are actually important, you should use L1. Otherwise, you should default to L2

# 3)The solver='sag' option in LogisticRegression and Ridge, which can be faster 
#   than the default on large datasets.
#   Other options are the SGDClassifier class and the SGDRegressor class  which implement even more 
#   scalable versions of the linear models 

# 4)Linear models often perform well when the number of features is large compared to
#   the number of samples. They are also often used on very large datasets, simply
#   because it’s not feasible to train other models.