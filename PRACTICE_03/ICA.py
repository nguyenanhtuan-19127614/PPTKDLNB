from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Independent Component Analysis
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X)
print(X_ica)

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_ica[y == i, 0], X_ica[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("Independent Component Analysis of IRIS dataset")

plt.show()
