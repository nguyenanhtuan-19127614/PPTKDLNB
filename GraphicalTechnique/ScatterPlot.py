import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('data.csv')
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
xs = np.random.rand(100)
ys = np.random.rand(100)
for x, y in zip(xs, ys):
   plt.scatter(x, y, cmap="copper")
plt.show()