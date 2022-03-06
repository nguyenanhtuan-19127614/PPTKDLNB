import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas

# Load datat từ file SalmonData.xlsx
data=pandas.read_csv("SalmonData.csv")
X_1=np.vstack((data["Freshwater_Alask"],data["Marine_Alask"])).T
X_2=np.vstack((data["Freshwater_Canadian"],data["Marine_Canadian"])).T
print(X_2)
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
S_pooled_inv=np.linalg.inv(S_pooled)
a=(mean_X_1-mean_X_2).T.dot(S_pooled_inv)
w=a.dot(mean_X_1-mean_X_2)*0.5

def classification_X(X,a,w):
    class_id=[]
    for i in X:
        print(w)
        if a.dot(i.T) >= w:
            class_id.append("Alask")
        else:
            class_id.append("Canadian")
    return np.array(class_id)

data_test=pandas.read_csv("Test.csv")
X_test=np.vstack((data_test["Freshwater"],data_test["Marine"])).T
y_test=data_test["Label"]

y_predict= classification_X (X_test,a,w)
print(y_predict)
accuracy_score_test= accuracy_score(y_predict,y_test)
print("accuracy score test: ",accuracy_score_test*100)


# plot
plt.plot(X_1[:,0], X_1[:,1], "o", color="blue",label="Alask")
plt.plot(X_2[:,0], X_2[:,1], "o", color="red",label="Canadian")
plt.legend()
plt.show()

temp_1=a.dot(X_1.T).reshape(-1,1)
temp_2=a.dot(X_2.T).reshape(-1,1)
plt.plot(temp_1,"o",color="blue",label="1")
plt.plot(temp_2,"o",color="red",label="0")
xx = np.arange(-2, 50, 0.01).reshape(-1, 1)
y=w+xx*0
plt.plot(xx,y,color="black")
plt.legend()
plt.show()