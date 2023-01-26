import pandas as pd
import numpy as np

def model(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((X.shape[1], 1))
    cost_list = []

    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred - Y))
        d_theta = (1/m)*np.dot(X.T, y_pred - Y)
         = theta - learning_rate*d_theta

        cost_list.append(cost)

        if(i % (iteration/10) == 0):
            print("Cost is: ", cost)
    return theta, cost_list


df = pd.read_csv('A2Q2Data_train.csv')
X_train = df.iloc[:, :100].values
y_train = df.iloc[:, 100].values
iteration = 100
learning_rate = 0.000005
theta, cost_list = model(X_train, y_train, learning_rate = learning_rate, iteration = iteration)
