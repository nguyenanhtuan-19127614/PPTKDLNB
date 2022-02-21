import numpy as np

# Mean
print("Mean-------------------------------------------------------------")
X = [1, 2, 3, 4, 5]
print("Mean X = ", np.mean(X))
X = np.array([[1, 2], [3, 4]])
print("Mean X = ", np.mean(X))
print("Mean X with axis = 0: ", np.mean(X, axis=0))
print("Mean X with axis = 1: ", np.mean(X, axis=1))
a = np.zeros((2, 512 * 512), dtype=np.float32)
a[0, :] = 1.0
a[1, :] = 0.1
print("a.shape: ", a.shape)
print("mean a = ", np.mean(a))
print("mean a = ", np.mean(a, dtype=np.float64))

# Median
print("Median-------------------------------------------------------------")
X = np.array([2, 5, 3, 1, 7])
Y = np.array([2, 1, 8, 5, 7, 9])
print("Median X = ", np.median(X))
print("Median Y = ", np.median(Y))
arr = np.array([[7, 4, 2], [3, 9, 5]])
print("median arr (axis = 0) = ", np.median(arr, axis=0))
print("median arr (axis = 1) = ", np.median(arr, axis=1))

# Mean & Median with NaN
x = np.array([2, np.nan, 5, 9])
print("Mean & Median with NaN-------------------------------------------------------------")
print("mean = ", np.mean(x))
print("median = ", np.median(x))
print("mean = ", np.nanmean(x))
print("median = ", np.nanmedian(x))

# Variance & Standard Deviation
print("Variance & Standard Deviation-------------------------------------------------------------")
X = [19, 33, 51, 22, 18, 13, 45, 24, 58, 11, 25, 27, 26, 29]
print("Variance: ", np.var(X))
print("Standard Deviation: ", np.std(X))
# Variance & Standard Deviation with NaN
print("Variance & Standard Deviation with NaN-------------------------------------------------------------")
A = np.array([1, np.nan, 3, 4])
print("var = ", np.var(A))
print("std = ", np.std(A))
print("nan var = ", np.nanvar(A))
print("nan std = ", np.nanstd(A))

# Order statistics with NaN
print("Order statistics with NaN-------------------------------------------------------------")
X = np.array([[14, 96],
 [np.nan, 82],
 [80, 67],
 [77, np.nan],
 [99, 87]])
print("X = ", X)
print("Max: ", np.nanmax(X))
print("Min: ", np.nanmin(X))

# Range
print("Range-------------------------------------------------------------")
X = np.array([[14, 96],
 [46, 82],
 [80, 67],
 [77, 91],
 [99, 87]])
print("x = ", X)
print("Range = ", np.ptp(X))
print("Range (axis = 0) = ", np.ptp(X, axis=0))
print("Range (axis = 1) = ", np.ptp(X, axis=1))