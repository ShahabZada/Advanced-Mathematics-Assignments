import csv
import matplotlib.pyplot as plt
import numpy as np
from math import pi
file = open('asteroid_data.csv')
csvreader = csv.reader(file)
x = []
y = []
vx,vy = 0,0
for row in csvreader:
    x.append(row[0])
    y.append(row[1])
file.close()
X = np.asfarray(x)
X = np.asfarray(x).reshape([X.size,1])
X = np.hstack((np.ones((X.size,1)), X))
Y = np.asfarray(y).reshape([X.shape[0],1])
val=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
fn=val[0] + val[1]*X[:,1]
plt.scatter(X[:,1], Y)
plt.plot(X[:,1],fn,'r')
plt.show()
