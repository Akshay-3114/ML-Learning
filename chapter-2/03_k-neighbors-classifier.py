#######################    An example of the k-NN classifier algorithm   ##########
import mglearn
import mglearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)   
print("Test set predictions: {}".format(clf.predict(X_test)))

print("Score: {:.2f}".format(clf.score(X_test, y_test)))


#       The following code produces the visualizations of the decision boundaries for one,
#        three, and nine neighbors shown


fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9, 26], axes):
 # the fit method returns the object self, so we can instantiate
 # and fit in one line
 fit_all_points = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
 mglearn.plots.plot_2d_separator(fit_all_points, X, fill=True, eps=0.5, ax=ax, alpha=.4)
 mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
 ax.set_title("{} neighbor(s)".format(n_neighbors))
 ax.set_xlabel("feature 0")
 ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
# print("1 count:- {}".format(sum(x==1 for x in y)))

#               A smoother boundary corresponds to a
#               simpler model. In other words, using few neighbors corresponds to high model complexity (as shown on the right side), 
#               and using many neighbors corresponds to low model complexity (as shown on the left side).


#               Thus when the n_neigbors = 1 (underfitted), as we go up with k the "Decision boundary" becomes smoother
#               and at the extreme case where the number of neighbors is the number of all datapoints(overfitted) in the training set, 
#               each test point would have exactly the same neighbors (all training points) and all predictions
#               would be the same: the class that is most frequent in the training set.




