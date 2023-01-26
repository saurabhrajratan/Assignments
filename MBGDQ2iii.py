from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X,y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# reg = LinearRegression()
# reg.fit(X_train,y_train)

# print(reg.coef_)
# print(reg.intercept_)

# y_pred = reg.predict(X_test)
# r2_score(y_test,y_pred)

import random

class MBGDRegressor:
    
    def __init__(self,batch_size,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.error = []
        
    def fit(self,X_train,y_train):
        # init your coefs
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        
        for i in range(self.epochs):
            
            for j in range(int(X_train.shape[0]/self.batch_size)):
                
                idx = random.sample(range(X_train.shape[0]),self.batch_size)
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_
                #print("Shape of y_hat",y_hat.shape)
                intercept_der = -2 * np.mean(y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx])
                self.coef_ = self.coef_ - (self.lr * coef_der)
            self.error.append(np.sqrt(np.sum(((np.dot(X_train,self.coef_) + self.intercept_)-y_train)**2)))
        
        print(self.intercept_,self.coef_)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_

    def plot(self):
        iterations = np.arange(1, len(self.error)+1, 1)
        print("len of iteratons: ", len(iterations))
        print("len of error: ", len(self.error))
        x = np.array(self.error)
        y = np.array(iterations)
        # x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
        # y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
        print("x: ", x.shape)
        print("y: ", y)
        plt.scatter(x, y)
        plt.show()    

df = pd.read_csv('A2Q2Data_train.csv')
X_train = df.iloc[:100, :100].values
y_train = df.iloc[:100, 100].values
mbr = MBGDRegressor(batch_size=int(X_train.shape[0]/100),learning_rate=0.01,epochs=100)
mbr.fit(X_train,y_train)
mbr.plot()

# df = pd.read_csv('A2Q2Data_test.csv')
# X_test = df.iloc[:, :100].values
# y_test = df.iloc[:, 100].values
# y_pred = mbr.predict(X_test)
# r2_score(y_test,y_pred)

# from sklearn.linear_model import SGDRegressor
# sgd = SGDRegressor(learning_rate='constant',eta0=0.1)
# batch_size = 35

# for i in range(100):
    
#     idx = random.sample(range(X_train.shape[0]),batch_size)
#     sgd.partial_fit(X_train[idx],y_train[idx])
# sgd.coef_
# sgd.intercept_
# y_pred = sgd.predict(X_test)
# r2_score(y_test,y_pred)
