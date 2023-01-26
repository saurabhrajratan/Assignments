from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time

np.random.seed(42)
# X,y = load_diabetes(return_X_y=True)
df = pd.read_csv('A2Q2Data_train.csv')
# print(df.shape)
X = df.iloc[:, :100].values
y = df.iloc[:, 100].values
# print("Type X_train: ", type(y_train))
# print("Type X: ", type(y))
# print("X_train: ", X_train.shape)
# print("y_train: ", y_train.shape)
# print("X.shape: ", X.shape)
# print("y.shape: ", y.shape)

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.2,random_state=2)
reg = LinearRegression()
reg.fit(X_train,y_train)

print("coef using Lin Reg: ", reg.coef_)
print("intercept using Lin Reg: ", reg.intercept_)

y_pred = reg.predict(X_test)
print("Score using Lin Reg: ", r2_score(y_test,y_pred))

class SGDRegressor:
    
    def __init__(self,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
        # init your coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                idx = np.random.randint(0,X_train.shape[0])
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_
                
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)
                
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
        
        print(self.intercept_,self.coef_)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_

sgd = SGDRegressor(learning_rate=0.01,epochs=40)

start = time.time()
sgd.fit(X_train,y_train)
print("The time taken is",time.time() - start)

y_pred = sgd.predict(X_test)

print("Score Using My SGDRegressor: ", r2_score(y_test,y_pred))

from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(max_iter=100,learning_rate='constant',eta0=0.01)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Score Using sklearn SGDRegressor: ",r2_score(y_test,y_pred))
