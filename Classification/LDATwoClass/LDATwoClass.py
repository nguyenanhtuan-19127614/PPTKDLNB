#Những thư viên cần thiết
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets
import matplotlib.pyplot as plt

# LDA theo công thức ---------------------------------------------------------
# Dataset
np.random.seed(22)# tạo dữ liệu random
mean0 = np.array([0, 5])
mean1= np.array([5, 0])
cov0 = np.array([[4, 3], [3, 4]])
cov1 = np.array([[3, 1], [1, 1]])
N0 = 50#số lượng phần tử của class0
N1 = 40#số lượng phần tử của class1
X0=np.random.multivariate_normal(mean0, cov0, N0)
X1=np.random.multivariate_normal(mean1, cov1, N1)
X = np.concatenate((X0, X1))# xếp chồng X0 và x1
y = np.array([0]*N0 + [1]*N1)#gán nhãn cho X0 và X1

# Build S_B (between-class)
m0 = np.mean(X0.T, axis = 1, keepdims = True)
m1 = np.mean(X1.T, axis = 1, keepdims = True)
a = (m0 - m1)
S_B = a.dot(a.T)

# Build S_W (within-class)
A = X0.T - np.tile(m0, (1, N0))
B = X1.T - np.tile(m1, (1, N1))
S_W = A.dot(A.T) + B.dot(B.T)

# Tìm vector riêng ứng với trị riêng lớn nhất
_, W = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
w = W[:,0]
print("Tự tính: ",w)
# LDA trogn thư viện scikit-learn ---------------------------------------------------------
clf=LDA()
clf.fit(X,y)#fit tập Dataset
w=clf.coef_/np.linalg.norm(clf.coef_)#vector w đã được chuẩn hóa
Y_predict=clf.predict(X)# predict X
print(Y_predict)
accuracy_score=metrics.accuracy_score(Y_predict,y)#tính đọ chính xác của tập
print("Theo skitlearn: ",w)
print(accuracy_score)

#plot data original
plt.plot(X0[:,0],X0[:,1],"o",color='red',label='Class 0')
plt.plot(X1[:,0],X1[:,1],"o",color='blue',label='Class 1')
plt.title("Original Data")
plt.legend()
plt.show()
#plot data with LDA
#
x = np.linspace(-10, 10, 1000)
plt.plot(X0[:,0],X0[:,1],"o",color='red',label='Class 0')
plt.plot(X1[:,0],X1[:,1],"o",color='blue',label='Class 1')
plt.plot(x,x*w[0,1]/w[0,0],label="Solution by LDA")
plt.title("LDA with 2 class")
plt.legend()
plt.show()