########################  Linear Regression Models    ##############################


# Linear models make a prediction using a linear function of the input features.
# For regression, the general prediction formula for a linear model looks as follows:
#
#     ŷ = w[0] * x[0] + w[1] * x[1] + ... + w[p] * x[p] + b
#
# Here, x[0] to x[p] denotes the features (in this example, the number of features is p)
# of a single data point, w and b are parameters of the model that are learned, and ŷ is
# the prediction the model makes. For a dataset with a single feature, this is:
#
#            ŷ = w[0] * x[0] + b

# which you might remember from high school mathematics as the equation for a line.
# Here, w[0] is the slope and b is the y-axis offset. For more features, w contains the
# slopes along each feature axis. Alternatively, you can think of the predicted response
# as being a weighted sum of the input features, with weights (which can be negative)
# given by the entries of w.

# Linear models for regression can be characterized as regression models for which the
# prediction is a line for a single feature, a plane when using two features, or a hyper‐
# plane in higher dimensions (that is, when using more features).


# Linear regression finds the parameters w and b that mini‐
# mize the mean squared error between predictions and the true regression targets, y,
# on the training set. The mean squared error is the sum of the squared differences
# between the predictions and the true values. Linear regression has no parameters,
# which is a benefit, but it also has no way to control model complexity


import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

# The “slope” parameters (w), also called weights or coefficients, are stored in the coef_
# attribute, while the offset or intercept (b) is stored in the intercept_ attribute.

# scikit-learn always stores anything
# that is derived from the training data in attributes that end with a
# trailing underscore


print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))


print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# An R^2 of around 0.66 is not very good, but we can see that the scores on the training
# and test sets are very close together. This means we are likely underfitting, not over‐
# fitting. For this one-dimensional dataset, there is little danger of overfitting, as the
# model is very simple (or restricted).

X_boston, y_boston = mglearn.datasets.load_extended_boston()
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_boston, y_boston, random_state=42
)

lr_b = LinearRegression().fit(X_b_train, y_b_train)
print("Training set score: {:.2f}".format(lr_b.score(X_b_train, y_b_train))) # 0.91
print("Test set score: {:.2f}".format(lr_b.score(X_b_test, y_b_test))) # 0.78
plt.show()


# This discrepancy between performance on the training set and the test set is a clear
# sign of overfitting, and therefore we should try to find a model that allows us to control complexity
