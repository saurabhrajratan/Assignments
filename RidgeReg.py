import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class MeraRidgeGD:
    
    def __init__(self,epochs,learning_rate,alpha):
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self,X_train,y_train):
        
        self.coef_ = np.ones((X_train.shape[1], 1))
        # self.coef_ = np.ones(X_train.shape[1])
        print("coef shape: ", self.coef_.shape)
        self.intercept_ = 0
        thetha = np.insert(self.coef_, 0, self.intercept_, axis=0)
        # print(thetha)
        
        X_train = np.insert(X_train,0,1,axis=1)
        print("shape of thetha: ", thetha.shape)
        print("shape of X: ", X_train.shape)
        # print("shape of Xw: ", np.matrix(np.dot(X_train, thetha).shape))
        print("shape of y: ", y_train.shape)
        for i in range(self.epochs):
            thetha_der = np.dot(X_train.T,X_train).dot(thetha) - np.dot(X_train.T,y_train) + self.alpha*thetha
            print((thetha - self.learning_rate*thetha_der).shape)
            # print("thetha shape: ",thetha.shape)
            thetha = thetha - self.learning_rate*thetha_der
        
        self.coef_ = thetha[1:]
        self.intercept_ = thetha[0]
    
    def predict(self,X_test):
        
        return np.dot(X_test,self.coef_) + self.intercept_

# Training
df = pd.read_csv('A2Q2Data_train.csv')
# print(df.shape)
X_train = df.iloc[:100, :100].values
y_train = df.iloc[:100, 100].values
# print("len X_train: ", len(X_train[0]))
# print("len y_train: ", len(y_train))
# print("X_train: ", X_train[0:3])
# print("y_train: ", y_train[0:3])
reg = MeraRidgeGD(epochs=500,alpha=0.001,learning_rate=0.005)
reg.fit(X_train,y_train)
print("coef_: ", reg.coef_)
print("intercept: ", reg.intercept_)

# Testing
# df = pd.read_csv('A2Q2Data_test.csv')
# X_test = df.iloc[:, :99].values
# y_test = df.iloc[:, 99:100].values
# y_pred = reg.predict(X_test)
# print("R2 score",r2_score(y_test,y_pred))