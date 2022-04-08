import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
#https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
#Load data
iris = datasets.load_iris()
# Xài panda để visualize data
# set max columns to none
pd.set_option("display.max_columns", None)
# print column without making newline
pd.set_option('display.expand_frame_repr', False)
df = pd.DataFrame(np.column_stack((iris.data, iris.target)), columns = iris.feature_names+['Species'])
print(df,end =" ")
#get data and target
X = iris.data
y = iris.target
target_names = iris.target_names
#Tính PCA
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
#tính LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
# print(
#     "explained variance ratio (first two components): %s"
#     % str(pca.explained_variance_ratio_)
# )
# input("Enter to continue...")

#PCA plot
plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")

#LDA plot
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_lda[y == i, 0], X_lda[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset")

plt.show()