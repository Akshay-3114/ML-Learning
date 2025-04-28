# Naive Bayes classifiers are a family of classifiers that are quite similar to the linear
# models discussed in the previous section. However, they tend to be even faster in
# training. This is at the trade-off with worser generalization.

# There are three kinds of naive Bayes classifiers implemented in scikit-learn
    # 1)Gaussian Naive Bayes(GaussianNB):
    #     -Used when features are continuous data (not discrete).
    #     Ex:- you are classifying fruit based on weight.

    # 2) Bernoulli Naive Bayes(BernoulliNB):
            # -Used when features are binary (0 or 1).
            # Ex:- Does email contain the word "Buy"? (Yes/No)

    # 3) Multinomial Naive Bayes(MultinomialNB):
            # - Used when features are counts.
            # Ex:- Classifying documents into Sports and Politics based on 
            #         the number of times related words appear



# Understanding BernoulliNB
import numpy as np

X = np.array([[0, 1, 0, 1],
 [1, 0, 1, 1],
 [0, 0, 0, 1],
 [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

# Here, we have four data points, with four binary features each. There are two classes,
# 0 and 1. For class 0 (the first and third data points), the first feature is zero two times
# and nonzero zero times, the second feature is zero one time and nonzero one time,
# and so on. These same counts are then calculated for the data points in the second class.

counts = {}
for label in np.unique(y):
 # iterate over each class
 # count (sum) entries of 1 per feature
 counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))



#Conclusion Notes

# 1)GaussianNB is mostly used on very high-dimensional data, while the other two variants 
#   of naive Bayes are widely used for sparse count data such as text. 

# 2)MultinomialNB usually performs better than BinaryNB, particularly on datasets with a relatively large
#   number of nonzero features (i.e., large documents).