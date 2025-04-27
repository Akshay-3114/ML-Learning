################### K - NN Regression Algorithm ##################
# The fundamentals of K neighbors are same here as well, but imstead of classifying, we predict the output based
#       on the closest datapoints to the datapoints we are predicting.

import mglearn
import matplotlib.pyplot as plt
import mglearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)

reg.fit(X_train, y_train)

print("Prediction :- {}".format(reg.predict(X_test)))
print("Score :- {:.2f}".format(reg.score(X_test, y_test)))


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg_plot = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg_plot.fit(X_train, y_train)
    ax.plot(line, reg_plot.predict(line))
    ax.plot(X_train, y_train, "^", c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, "v", c=mglearn.cm2(1), markersize=8)
    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors,
            reg_plot.score(X_train, y_train),
            reg_plot.score(X_test, y_test),
        )
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(
    ["Model predictions", "Training data/target", "Test data/target"], loc="best"
)
plt.show()
