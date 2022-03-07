import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import pandas
#Data: https://github.com/rocksaint/fruit-data-with-colours
# Load dữ liệu
data=pandas.read_csv("Fruit.csv")
#query dữ liệu apple, orangle, lemon
apple = data.query('fruit_label==1')
orange = data.query('fruit_label==3')
lemon = data.query('fruit_label==4')

X_1=np.vstack((apple["mass"],apple["color_score"])).T
X_2=np.vstack((orange["mass"],orange["color_score"])).T
X_3=np.vstack((lemon["mass"],lemon["color_score"])).T
#loai bo NaN
X_1=X_1[~np.isnan(X_1).any(axis=1)]
X_2=X_2[~np.isnan(X_2).any(axis=1)]
X_3=X_3[~np.isnan(X_3).any(axis=1)]

#Lấy thông tin kích thước dữ liệu và số chiều dữ liệu
n_1=X_1.shape[0]
n_2=X_2.shape[0]
n_3=X_3.shape[0]

p_1=X_1.shape[1]
p_2=X_2.shape[1]
p_3=X_3.shape[1]

#tính priori probability
temp=n_1+n_2+n_3
priori_prob_1=1.0*n_1/temp
priori_prob_2=1.0*n_2/temp
priori_prob_3=1.0*n_3/temp
priori_prob=np.array([priori_prob_1,priori_prob_2,priori_prob_3])

# Tính sample mean
mean_X_1=np.mean(X_1,axis=0).reshape((p_1,1))
mean_X_2=np.mean(X_2,axis=0).reshape((p_2,1))
mean_X_3=np.mean(X_3,axis=0).reshape((p_3,1))
mean=np.array([mean_X_1,mean_X_2,mean_X_3]) #load vào array xử lý cho dễ

# Tính sample covariance matrix
S_1=np.cov(X_1.T)
S_2=np.cov(X_2.T)
S_3=np.cov(X_3.T)
# Tính S_pooled
S_pooled=(1/(n_1+n_2+n_3-3))*((n_1-1)*S_1+(n_2-1)*S_2+(n_3-1)*S_3)
# nghịch đảo S_pooled
S_pooled_inv=np.linalg.inv(S_pooled)

#Hệ số của tổ hợp tuyến tính
a_1=mean_X_1.T.dot(S_pooled_inv)[0]
a_2=mean_X_2.T.dot(S_pooled_inv)[0]
a_3=mean_X_3.T.dot(S_pooled_inv)[0]
a=np.array([a_1,a_2,a_3])

def discriminant_score(X,priori_prob,a,mean_X): # tính discriminant score
    return np.log(priori_prob)+a.dot(X)-0.5*a.dot(mean_X)

def classification(X,priori_prob,a,mean): # Phân lớp dựa trên d lớn nhất
    class_id=[]
    for i in X:
        #tính d (discriminant score)
        d_1=discriminant_score(i,priori_prob[0],a[0],mean[0])
        d_2=discriminant_score(i,priori_prob[1],a[1],mean[1])
        d_3=discriminant_score(i,priori_prob[2],a[2],mean[2])

        #Lựa discriminant score lớn nhất, x sẽ thuộc lớp có  discriminant score lớn nhất này
        if d_1>d_2 and d_1>d_3:
            class_id.append("apple")
        if d_2>d_1 and d_2>d_3:
            class_id.append("orange")
        if d_3>d_1 and d_3>d_2:
            class_id.append("lemon")
    return np.array(class_id)

#Test model
data_test=pandas.read_csv("Test.csv")
X_test=np.vstack((data_test["mass"],data_test["color_score"])).T
y_test=data_test["result"]

y_predict= classification (X_test,priori_prob,a,mean)
print(y_predict)
accuracy_score_test= accuracy_score(y_predict,y_test)
print("accuracy score test: ",accuracy_score_test*100)
#plot
plt.plot(X_1[:,0], X_1[:,1],"o", color="red",label="Apple")
plt.plot(X_2[:,0], X_2[:,1], "o",color="orange",label="Orange")
plt.plot(X_3[:,0], X_3[:,1], "o",color="yellow",label="Lemon")
plt.xlabel("Mass")
plt.ylabel("Color Score")
plt.title("Dataset")
plt.legend()
plt.show()

# Plot data test
class_1=X_test[y_test=="apple"]
class_2=X_test[y_test=="orange"]
class_3=X_test[y_test=="lemon"]
plt.plot(class_1[:,0], class_1[:,1], "o",color="red",label="Apple")
plt.plot(class_2[:,0], class_2[:,1], "o",color="orange",label="Orange")
plt.plot(class_3[:,0], class_3[:,1], "o",color="yellow",label="Lemon")
plt.xlabel("Mass")
plt.ylabel("Color Score")
plt.title("Data test")
plt.legend()
plt.show()

# Plot data test discriminant_score compare
d_apple=[]
d_orange=[]
d_lemon=[]
for i in X_test:
    # tính d (discriminant score)
    d_1 = discriminant_score(i, priori_prob[0], a[0], mean[0])
    d_2 = discriminant_score(i, priori_prob[1], a[1], mean[1])
    d_3 = discriminant_score(i, priori_prob[2], a[2], mean[2])
    d_apple.append(d_1)
    d_orange.append(d_2)
    d_lemon.append(d_3)
plt.plot(d_apple, "o",color="red",label="Apple")
plt.plot(d_orange, "o",color="orange",label="Orange")
plt.plot(d_lemon, "o",color="yellow",label="Lemon")
plt.xlabel("Items")
plt.ylabel("Discriminant Score")
plt.title("Fruit discriminant_score compare")
plt.legend()
plt.show()
