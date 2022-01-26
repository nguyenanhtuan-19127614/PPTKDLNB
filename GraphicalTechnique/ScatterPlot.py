import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10]
y = [50,48,45,40,35,30,20,15,14,11]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x, y)
ax.set(title = "Scatter Plot Example",
       xlabel = "Employees",
       ylabel = "Profits per employee")
plt.show()