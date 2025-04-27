import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import mglearn


def custom_colormap():
    from matplotlib.colors import ListedColormap

    return ListedColormap([(0, 0, 1), (1, 0, 0), (0, 1, 0)])


iris_data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_data["data"], iris_data["target"], random_state=0
)

df = pd.DataFrame(X_train, columns=iris_data["feature_names"])
# plt.figure(figsize=(15, 15))
try:
    grr = pd.plotting.scatter_matrix(
        df,
        c=y_train,  # color based on target
        figsize=(15, 15),
        marker="o",
        hist_kwds={"bins": 20},
        s=60,  # marker size
        alpha=0.8,  # transparency
        cmap=mglearn.cm3,  # custom colormap to replace mglearn.cm3
    )

    # plt.suptitle("Iris Dataset Scatter Matrix", fontsize=16)
    # plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")
