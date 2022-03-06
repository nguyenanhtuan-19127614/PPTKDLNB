import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas

# Load datat từ file SalmonData.xlsx
data=pandas.read_csv("SalmonData.csv")
X_1=np.vstack((data["Freshwater_Alask"],data["Marine_Alask"])).T
X_2=np.vstack((data["Freshwater_Canadian"],data["Marine_Canadian"])).T

n_1=X_1.shape[0]
n_2=X_2.shape[0]
p_1=X_1.shape[1]
p_2=X_2.shape[1]

# Tính sample mean
mean_X_1=np.mean(X_1,axis=0).reshape((p_1,1))
mean_X_2=np.mean(X_2,axis=0).reshape((p_2,1))
# Tính covariance matrix
S_1=np.cov(X_1.T)
S_2=np.cov(X_2.T)

# Tính S_pooled
S_pooled=(1/(n_1+n_2-2))*((n_1-1)*S_1+(n_2-1)*S_2)
# S_pooled nghịch đảo
S_pooled_inv=np.linalg.inv(S_pooled)
# a= chuyển vị (mean vector x1 - mean vector x2) X S_pooled nghịch đảo
a=(mean_X_1-mean_X_2).T.dot(S_pooled_inv)
#Tính m mũ
m=0.5*a.dot(mean_X_1-mean_X_2)

def classification_X(X,a,m):
    class_id=[]
    for i in X:
        print(a.dot(i.T))
        if a.dot(i.T) >= m:
            class_id.append("Alask")
        else:
            class_id.append("Canadian")
    return np.array(class_id)

data_test=pandas.read_csv("Test.csv")
X_test=np.vstack((data_test["Freshwater"],data_test["Marine"])).T
y_test=data_test["Label"]

y_predict= classification_X (X_test,a,m)
print(y_predict)
accuracy_score_test= accuracy_score(y_predict,y_test)
print("accuracy score test: ",accuracy_score_test*100)


# plot
plt.plot(X_1[:,0], X_1[:,1], "o", color="blue",label="Alask")
plt.plot(X_2[:,0], X_2[:,1], "o", color="red",label="Canadian")
plt.legend()
plt.show()

#Tính y của X_1 và X_2
y_1=a.dot(X_1.T).reshape(-1,1)
y_2=a.dot(X_2.T).reshape(-1,1)

plt.plot(y_1,"o",color="blue",label="Alask")
plt.plot(y_2,"o",color="red",label="Canadian")
xx = np.arange(-2, 50, 0.1).reshape(-1, 1)

m_new=m+xx*0 #m_new vẫn là m nhưng nhân với xx để tạo array vẽ cho dễ

plt.plot(xx,m_new,color="black")
plt.legend()
plt.show()