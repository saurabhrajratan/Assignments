from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

df=pd.read_csv("A2Q2Data_train.csv")

batch_size=100
def BatchStochasticDerivative(A,X,B):
    batch_count=0
    derivat_list=[]
    for i in range(0,len(df),batch_size):
        batch_count = batch_count + 1
        error=A[i:i+100,:]*X-B[i:i+100,:]
        derivat_sum=np.sum(error)
        derivative=[]
        for k in range(len(X)):
            derivative.append(2*X[k,0]*derivat_sum/batch_size)
        derivative=np.matrix(derivative).reshape(100,1)
        derivat_list.append(derivative)
    return sum(derivat_list)/batch_count

X=np.random.rand(100,1)
A=np.matrix(df.iloc[:,:100])
B=np.matrix(df.iloc[:,100]).reshape(9999,1)
epoch=1000
alpha=0.01

for i in tqdm(range(epoch)):
    gradient=BatchStochasticDerivative(A,X,B)
    X=X-(alpha*gradient)
print(X)

W_ML=X

X=np.random.rand(100,1)
L2_norm=[]
for i in tqdm(range(epoch)):
    gradient=BatchStochasticDerivative(A,X,B)
    X=X-(alpha*gradient)
    values=np.linalg.norm(X-W_ML,ord=2)
    L2_norm.append(values)

print("L2_norm: ", L2_norm)

fig = plt.figure(figsize=(8,5))
t=[i+1 for i in range(epoch)]
ax = fig.add_subplot(111, projection='3d')
plt.title('Plot for Error v/s Iterations of Batch Stochastic Gradient Descent Algorithm')
plt.xlabel("Error Axis ----->")
plt.ylabel("No. of Iterations Axis ----->")
ax.scatter(0,t,L2_norm)
plt.show()





