# For LogisticRegression and LinearSVC the trade-off parameter that determines the
# strength of the regularization is called C. When higher value of C(weaker regularization) is used, we try to fit the 
# training set as best as possible, while lower value of C(stronger regularization) corresponds to finding the coefficient vector(w)
# close to 0.

# Below code displays the example of same.


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
import mglearn.datasets
import matplotlib.pyplot as plt

# mglearn.plots.plot_linear_svc_regularization()


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer["data"], cancer["target"], stratify=cancer["target"], random_state=42)
lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)
print("Training set score: {:.3f}".format(lr.score(X_train, y_train))) #0.946
print("Test set score: {:.3f}".format(lr.score(X_test, y_test))) #0.958

# Here c = 1(Default value), and the test score, train score are very close. So we are likely underfitting.

logreg100 = LogisticRegression(C=100, max_iter=10000).fit(X_train, y_train)
print("Training set score for c = 100: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score for c = 100: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01, max_iter=10000).fit(X_train, y_train)
print("Training set score for c = 0.01: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score for c = 0.01: {:.3f}".format(logreg001.score(X_test, y_test)))

plt.plot(lr.coef_.T, 'o', label="C=1") # gives the coeffiecient vector(w) for each feature
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()


#we desire a more interpretable model, using L1 regularization might help, as it lim‐
# its the model to using only a few features. The example of the same would be :-

for C, marker in zip([0.01, 1, 100], ['o', '^', 'v']):
    # The penalty parameter controls what type of regularization you apply to the model.
    # 'solvers' — meaning optimization algorithms that help train your model by finding the best weights.
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver="liblinear").fit(X_train, y_train)  
    print("Training set score for c = {:.3f}: {:.3f}".format(C, lr_l1.score(X_train, y_train)))
    print("Test set score for c = {:.3f}: {:.3f}".format(C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))


plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()
