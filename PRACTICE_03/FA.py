from sklearn.decomposition import FactorAnalysis
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

#Factor Analysis
transformer = FactorAnalysis(n_components=2, random_state=0)
X_fa = transformer.fit_transform(X)
print(X_fa)

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_fa[y == i, 0], X_fa[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("Factor Analysis of IRIS dataset")

plt.show()