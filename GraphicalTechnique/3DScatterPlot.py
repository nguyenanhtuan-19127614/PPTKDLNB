import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('LizardSize.csv')

Mass=data['Mass']
SVL=data['SVL']
HLS=data['HLS']

fig = plt.figure()
ax = plt.axes(projection='3d')

my_cmap = plt.get_cmap('hsv')
ax.scatter3D(SVL, HLS, Mass)

ax.set(
    title="Lizard Size Data",
    xlabel='SVL',
    ylabel='HLS',
    zlabel='Mass',

)
plt.show()