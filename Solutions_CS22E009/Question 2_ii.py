import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_csv("A2Q2Data_train.csv")
# np.random.seed(42)
def partialDerivat(first,mid,last):
    p_derivative=[]
    derivat_sum=np.sum(first*mid-last)
    for i in range(len(X)):
        p_derivative.append(2*mid[i,0]*derivat_sum/9999)
    p_derivative=np.matrix(p_derivative).reshape(100,1)
    return p_derivative

X=np.random.rand(100,1)
X_train=np.matrix(df.iloc[:,:100])
y_train=np.matrix(df.iloc[:,100]).reshape(9999,1)
epoch=1000
alpha=0.001

for itr in range(epoch):
    gradient=partialDerivat(X_train,X,y_train)
    X=X-(alpha*gradient)
W_ML=X
print("W_ML : \n", W_ML)
X=np.random.rand(100,1)
L2_norm=[]
for itr in range(epoch):
    gradient=partialDerivat(X_train,X,y_train)
    X=X-(alpha*gradient)
    values=np.linalg.norm(X-W_ML,ord=2)
    L2_norm.append(values)

fig = plt.figure(figsize=(8,5))
t=[i+1 for i in range(epoch)]
ax = fig.add_subplot(111, projection='3d')
plt.title('Plot for Error v/s Iterations of Gradient Descent Algorithm')
plt.xlabel("Error Axis ----->")
plt.ylabel("No. of Iterations Axis ----->")
ax.scatter(0,t,L2_norm)
plt.show()

### OBSERVATION drawn from the plot of W-W_ML versus iteration : difference gradually decreases with each iteration of algorithm and become constant apparantly.

