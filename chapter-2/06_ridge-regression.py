#############################   Ridge Regression    #############################

# In linear regression, we try to find weights (coefficients) that minimize the error between predicted values and actual values.
# But sometimes, especially when:

# --> You have too many features

# --> Or your features are correlated

# --> Or you have very few data points

# Then linear regression tries too hard to fit the training data — this is called overfitting.
# It performs great on training data but badly on new/unseen data.

# Ridge Regression adds a penalty for large weights. Instead of just minimizing the error, it minimizes:
# Loss = Error + λ ⋅ (sum of squares of the weights)

# Or in mathematical form:

# Loss = ∑(Yᵢ − Ŷᵢ)² + λ ∑wⱼ²

# Where:

# Yᵢ: actual value
# Ŷᵢ: predicted value
# wⱼ: each model weight (excluding the bias term)
# λ: regularization strength (you control this)

# This constraint is an example of what is called regularization.
# Regularization is a technique used to prevent overfitting by discouraging the model from relying too much
# on any one feature or weight.
#           Or
# Regularization means explicitly restricting a model to avoid overfitting


from sklearn.linear_model import Ridge
import mglearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


X_boston, y_boston = mglearn.datasets.load_extended_boston()
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_boston, y_boston, random_state=42
)
ridge = Ridge().fit(X_b_train, y_b_train)
print("Train score : {:.2f}".format(ridge.score(X_b_train, y_b_train)))  # 0.87
print("Test score : {:.2f}".format(ridge.score(X_b_test, y_b_test)))  # 0.81


# How much importance the model places on simplicity versus training set performance can be specified by the
# user, using the alpha parameter. (Default alpha = 1.0)

# i) Higher alpha → more regularization → weights get smaller
# ii) Lower alpha → less regularization → behaves more like plain linear regression

ridge10 = Ridge(alpha=10).fit(X_b_train, y_b_train)
print(
    "Training set score for alpha 10: {:.2f}".format(
        ridge10.score(X_b_train, y_b_train)
    )
)  # 0.77
print(
    "Test set score for alpha 10: {:.2f}".format(ridge10.score(X_b_test, y_b_test))
)  # 0.73


ridge01 = Ridge(alpha=0.1).fit(X_b_train, y_b_train)
print(
    "Training set score for alpha 0.1: {:.2f}".format(
        ridge01.score(X_b_train, y_b_train)
    )
)  # 0.92
print(
    "Test set score for alpha 0.1: {:.2f}".format(ridge01.score(X_b_test, y_b_test))
)  # 0.82


plt.plot(ridge.coef_, "s", label="Ridge alpha=1")
plt.plot(ridge10.coef_, "^", label="Ridge alpha=10")
plt.plot(ridge01.coef_, "v", label="Ridge alpha=0.1")
# plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
# plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()
mglearn.plots.plot_ridge_n_samples()

# With enough training data, regularization becomes less important, and given enough data, ridge and
# linear regression will have the same performance
