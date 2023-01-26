from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df=pd.read_csv("A2Q2Data_train.csv",)

df.columns=[i for i in range(len(df.columns))]



#Closed form solution of Least Sqaure Error Of Regression Problem using Analytical Method
#W_ML=inverse(trans(A)A)*trans(A)*B
A=np.matrix(df.iloc[:,:100])
B=np.matrix(df.iloc[:,100]).reshape(9999,1)

A_transpo=A.transpose()

W_ML=np.linalg.inv(A_transpo*A)*A_transpo*B

print("W_ML: \n", W_ML)

df_test=pd.read_csv("A2Q2Data_test.csv",)
x_test=df_test.iloc[:,:100]
y_test=df_test.iloc[:,100]

ss_error=np.linalg.norm((np.matrix(x_test)*W_ML)-np.matrix(y_test).reshape(499,1))

print("SS Error : ",ss_error)

# class OrdLeastSquare:

#     def __init__(self):
#         self.coef_ = None
#         self.intercept_ = None
        
#     def modelFit(self,X_train,y_train):
#         X_train = np.insert(X_train,0,1,axis=1)
        
#         # calcuate the coeffs
#         omega = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
#         self.intercept_ = omega[0]
#         self.coef_ = omega[1:]
#         return self.coef_, self.intercept_
    
#     def predict(self, X_test):
#         y_pred = np.dot(X_test, self.coef_) + self.intercept_
#         return y_pred

# # Training
# df = pd.read_csv('A2Q2Data_train.csv')
# X_train = df.iloc[:, :99].values
# y_train = df.iloc[:, 99:100].values
# lr = OrdLeastSquare()
# lr.modelFit(X_train,y_train)
# print("coeff W_ML: ", lr.coef_)
# print("intercept: ", lr.intercept_)
# print("No. of weight vector : ", len(lr.coef_))

# # Testing
# df = pd.read_csv('A2Q2Data_test.csv')
# X_test = df.iloc[:, :99].values
# y_test = df.iloc[:, 99:100].values
# y_pred = lr.predict(X_test)
# # print("r2_score: ", r2_score(y_test,y_pred))
# # print(lr.coef_)


