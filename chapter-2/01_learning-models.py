# Chapter - 2
import mglearn
import mglearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# X, y = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(["Class 0", "Class 1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# plt.show()

# X_wave, y_wave = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X_wave, y_wave, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.show()

# X, y = mglearn.datasets.load_extended_boston()
# print("X: {}".format(y))


##########################      Supervised learning         #########################

#       The user provides the algorithm with pairs of
#       inputs and desired outputs, and the algorithm finds a way to produce the desired out‐
#       put given an input. In particular, the algorithm is able to create an output for an input
#       it has never seen before without any help from a human.

#       Machine learning algorithms that learn from input/output pairs are called supervised
#       learning algorithms because a “teacher” provides supervision to the algorithms in the
#       form of the desired outputs for each example that they learn from.

#       Example:- 1)Determining whether a tumor is benign based on a medical image
#                 2)Detecting fraudulent activity in credit card transactions


#       There are two major types of supervised machine learning problems, called classification
#       and regression.


#       Classification :- Here,  the goal is to predict a class label, which is a choice from a predefined
#                            list of possibilities.

#       Classification is sometimes separated into binary classification,
#       which is the special case of distinguishing between exactly two classes, and multiclass
#       classification, which is classification between more than two classes.

#       Regression :- Here, the goal is to predict a continuous number, or a floating-point
#                   number in programming terms (or real number in mathematical terms).
#            Ex:- i) A person’s annual income from their education, their age, and where they live.
#                 ii) Predicting the yield of a corn farm given attributes such as previous yields, weather, 
#                   and number of employees working on the farm.


#       An easy way to distinguish between classification and regression tasks is to ask whether 
#       there is some kind of continuity in the output. If there is continuity between possible outcomes, 
#       then the problem is a regression problem.




#################################################### Algorithms #############################################


####################    1) k - NN Algorithm      ################

#####   Textbook refernce    ######
#           The k-NN algorithm is arguably the simplest machine learning algorithm. Building
#           the model consists only of storing the training dataset. To make a prediction for a
#            new data point, the algorithm finds the closest data points in the training dataset—its
#           “nearest neighbors.”


#####   My understanding   ######
#            A simple algorithm that finds the closest(neighbors) to data point in the training set,
#            to make the prediction.

#            Depending on the value supplied to it using n_neigbors(k) parameter, we can control how many
#            closest datapoints(neigbors) the algorithm uses to make the prediction to the datapoint we have provided.

#            It is a classification algorithm.

#            Based on the number of closest neigbors a datapoint has, the datapoint will be classified via a voting system.(Will be put into a class which has the most number of neigbors to the datapoint).
#            It is recommended to have odd n_neighbors(k) so that the equal number of neigbors scenario doesn't occur
#            i.e., Use n_neigbors = 2 in the below code to see the effect.

#            For more classes, we count how many
#            neighbors belong to each class and again predict the most common class.

# Classification example
mglearn.plots.plot_knn_classification(n_neighbors=4)

# Regression Example
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()

