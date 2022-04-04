import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# data set random
np.random.seed(22)
mean0 = np.array([0, 5])
mean1= np.array([5, 0])
mean2= np.array([4,-5])
cov0 = np.array([[4, 3], [3, 4]])
cov1 = np.array([[3, 1], [1, 1]])
cov2 = np.array([[5, 1], [1, 2]])
N0 = 50#số lượng phần tử của class0
N1 = 40#số lượng phần tử của class1
N2 = 70#số lượng phần tử của class2
X0=np.random.multivariate_normal(mean0, cov0, N0)
X1=np.random.multivariate_normal(mean1, cov1, N1)
X2=np.random.multivariate_normal(mean2, cov2, N2)
X = np.concatenate((X0, X1,X2))# xếp chồng X0 và x1
y = np.array([0]*N0 + [1]*N1+[2]*N2)#gán nhãn cho X0 và X1

# LDA trong thư viện scikit-learn
clf=LDA()
clf.fit(X,y)#fit tập Dataset
w=clf.coef_/np.linalg.norm(clf.coef_)#vector w đã được chuẩn hóa
Y_predict=clf.predict(X)# predict X
accuracy_score=metrics.accuracy_score(Y_predict,y)#tính đọ chính xác của tập traning
print("accuracy_score: ",accuracy_score*100,"%")
print("w: ",w)

#Plot original data
x = np.linspace(-15, 15, 1000)
plt.plot(X0[:,0],X0[:,1],"o",color='red',label='Class 0')
plt.plot(X1[:,0],X1[:,1],"o",color='blue',label='Class 1')
plt.plot(X2[:,0],X2[:,1],"o",color='green',label='Class 2')
plt.title("Original Data of 3 class")
plt.legend()
plt.show()

#Plor LDA
X_set, y_set = X, y
aranged_pc1 = np.arange(start=X_set[:, 0].min(), stop=X_set[:, 0].max(), step=0.01)
aranged_pc2 = np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max(), step=0.01)
X1, X2 = np.meshgrid(aranged_pc1, aranged_pc2)
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
    alpha=0.5, cmap=ListedColormap(('orange', 'blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('LDA for 3 class')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()