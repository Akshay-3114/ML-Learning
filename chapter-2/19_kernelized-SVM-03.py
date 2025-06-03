# Preprocessing data for SVMs

# One way to resolve the effects of feature magnitude is by rescaling.
# A common rescaling method for kernel SVMs is to scale the data such that all features are between 0 and 1.

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0
)
# compute the minimum value per feature on the training set
min_on_training = X_train.min(axis=0)
# compute the range of each feature (max - min) on the training set
range_on_training = (X_train - min_on_training).max(axis=0)
# subtract the min, and divide by range
# afterward, min=0 and max=1 for each feature
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))

X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC().fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
 svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))


#--------------------------------Strengths, weaknesses, and parameters--------------------------------------#

# --->  SVMs allow for complex decision boundaries, even if the data has only a few features. 
#       They work well on low-dimensional and high-dimensional data.

# --->  Running an SVM on data with up to 10,000 samples might work well, but working with 
#       datasets of size 100,000 or more can become challenging in terms of runtime and memory usage.

# --->  They require careful preprocessing of the data and tuning of the parameters and are hard to inspect.
        #  it can be difficult to understand why a particular prediction was made, and it might be tricky
        #  to explain the model to a nonexpert.

# ---> Good settings for the two parameters are usually strongly correlated, and C and gamma 
#        should be adjusted together.