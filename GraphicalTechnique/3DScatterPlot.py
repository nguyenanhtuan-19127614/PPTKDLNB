import matplotlib.pyplot as plt
import numpy as np

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin
Mass=[5.526,10.401,9.213,8.953,7.063,6.610,11.273]
SVL=[59.0,75.0,69.0,67.5,62.0,62.0,74.0]
HLS=[113.5,142.0,124.0,125.0,129.5,123.0,140.0]

fig = plt.figure()
ax = plt.axes(projection='3d')


my_cmap = plt.get_cmap('hsv')
sctt = ax.scatter3D(SVL, HLS, Mass,
                    alpha = 0.8,)
# for i in range(len(Mass)):
#     xs = HLS[i]
#     ys = SVL[i]
#     zs = Mass[i]
#     ax.scatter(xs, ys, zs, marker=i)

ax.set(
    title="Lizard Size Data",
    xlabel='SVL',
    ylabel='HLS',
    zlabel='Mass',

)
plt.show()