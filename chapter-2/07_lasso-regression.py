##########################################   Lasso Regression    ###########################################

# An alternative to Ridge for regularizing linear regression is Lasso. As with ridge
# regression, using the lasso also restricts coefficients to be close to zero, but in a
# slightly different way, called L1 regularization.

# The consequence of L1 regularization is that when using the lasso, some coefficients are exactly zero.

# Loss = ∑(Yi - Ŷi)² + λ * ∑(|wj|), same terms as in ridge

# “If this feature isn’t helping — I’ll drop its weight to zero and toss it away.”
# (acts like a built-in feature selector)


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import mglearn.datasets
import matplotlib.pyplot as plt
import numpy as np


X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# underfitting
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))  # 0.27
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))  # 0.26
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))  # 3

# We are underfitting since Lasso does bad for both training and test data set.
# It has a default regularization strenght(alpha) of 1.0

# generalized
# The maximum number of iterations the solver (optimizer) will run to try and find the best weights(max_iter).
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)


print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

# overfitting
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))


plt.plot(lasso.coef_, "s", label="Lasso alpha=1")
plt.plot(lasso001.coef_, "^", label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, "v", label="Lasso alpha=0.0001")
# plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()


# Use Ridge when:
# → All features are useful, or you have many correlated features

# Use Lasso when:
# → You want feature selection, or you suspect some features are junk


# scikit-learn also provides
# the ElasticNet class, which combines the penalties of Lasso and Ridge. In practice,
# this combination works best, though at the price of having two parameters to adjust:
# one for the L1 regularization, and one for the L2 regularization
