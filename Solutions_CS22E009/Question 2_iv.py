import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

df=pd.read_csv("A2Q2Data_train.csv")

def partialDerivat(A,X,B,rp_lambda):
    derivative_list=[]
    for i in range(len(X)):
        error=A*X-B
        error_mat=np.transpose(A[:,i])*error
        derivat_sum=np.sum(error_mat)
        derivative_list.append(2*derivat_sum+2*rp_lambda*int(X[i,0]))
    derivative=np.matrix(derivative_list).reshape(100,1)
    return derivative       

#Generationg random value of Regularization Parameter lambda
reg_para_lambda=np.linspace(0,0.3,5)
epoch=100
alpha=0.1
lambda_error=[]
X_list=[]
# Cross Validation Step
for l in reg_para_lambda:
    df_c=df.sample(frac=1)
    X=np.random.rand(100,1)
    A_train,A_test=np.matrix(df_c.iloc[:8000,:100]),np.matrix(df_c.iloc[8000:,:100])
    B_train,B_test=np.matrix(df_c.iloc[:8000,100]), np.matrix(df_c.iloc[8000:,100])
    for i in tqdm(range(epoch)):
        gradient=partialDerivat(A_train,X,B_train,l)
        X=X-(alpha*gradient)
    X_list.append(X)
    lambda_error.append(np.linalg.norm(A_test*X-B_test, ord=2))
index=np.argmin(lambda_error)
W_R=X_list[index]
lambda_optimal=reg_para_lambda[index]
print("W_r : ",W_R)
print("lambda : ",lambda_optimal)

plt.plot(reg_para_lambda,lambda_error,color="Red")
plt.title("Plot for Gradient Descent Algorithm for Ridge Regression")
plt.xlabel("Regularization Constant (Lambda) Axis ----->")
plt.ylabel("Validation Error Axis ----->")
plt.show()

df_test=pd.read_csv("A2Q2Data_test.csv")
df_test.columns=[i for i in range(101)]

x_test=df_test.iloc[:,:100]
y_test=df_test.iloc[:,100]

# Exploiting the SS Error obtained in previous Question
Error_w_ml= 13.614527124310232
Error_w_r=np.linalg.norm((np.matrix(x_test)*W_R)-np.matrix(y_test).reshape(499,1))
print("Error_w_r: ", Error_w_r)

### OBSERVATION drawn from the plot of Gradient Descent Algo for Ridge Regression : Due to Regularization Parameter, Ridge Regression better converges with lower error

