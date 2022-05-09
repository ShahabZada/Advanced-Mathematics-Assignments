import csv
import matplotlib.pyplot as plt
import numpy as np

from computeCost import *
from gradientDescent import *

file = open('student_debt.csv')
csvreader = csv.reader(file)
#header = next(csvreader)
#print(header)
#rows = []
years = []
debt = []

theta = np.zeros((2, 1))

m = 10
c = 0
for row in csvreader:
    #print(row[1])
    #rows.append(row)
    years.append(row[0])
    debt.append(row[1])
#print(rows)
file.close()

#print(years)



X = np.asfarray(years)
print("X.size ",X.size)
X = np.asfarray(years).reshape([X.size,1])
print("X.sizeNp ",X.shape)
X = np.hstack((np.ones((X.size,1)), X))
print("Xshape>>>>>> = ",X.shape)

#X = np.expand_dims(X, axis=1)
#X_data= np.ones((X.shape[0],1))
#X_data=np.append(X_data, X, axis = 1)
#print(type(X[0]))
Y = np.asfarray(debt).reshape([X.shape[0],1])
print("Yshape = ",Y.shape)
#print("cost = ",computeCost(X,Y,m,c))
print("xshape",X.shape,"  x.tshape", X.T.shape)
theta=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
print("test = ",theta)
""""
alpha = 0.0000000000025
num_iters = 30

[theta,cost] = gradientDescent(X, Y, theta, alpha, num_iters)
print(cost)
"""
fn=theta[0] + theta[1]*X[:,1]
print("fn=",fn)
print("theta=", theta)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1],fn,'r')
plt.xlabel('Years')
plt.ylabel('Debt')
plt.legend(["Observations", "Linear regression model"], loc ="lower right")
plt.show()


#######################