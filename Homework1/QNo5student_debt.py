import csv
import matplotlib.pyplot as plt
import numpy as np

##################################################################
################    Question No 5 Solution            ############
##################################################################

#reading csv file

file = open('student_debt.csv')
csvreader = csv.reader(file)
years = []
debt = []

#variable to be optimized
theta = np.zeros((2, 1))

for row in csvreader:
    years.append(row[0])
    debt.append(row[1])
file.close()

#converting list of strings to numpy float array
X = np.asfarray(years)
#print("X.size ",X.size)
X = np.asfarray(years).reshape([X.size,1])
#print("X.sizeNp ",X.shape)
#stack ones to X
X = np.hstack((np.ones((X.size,1)), X))
#print(X)

Y = np.asfarray(debt).reshape([X.shape[0],1])
#print("Yshape = ",Y.shape)

#########################################################################
############           Question No 5 Part (a)               #############
#########################################################################

#########     equation for variable optimization
#########     x_hat = ((A'A)^-1)A'y
#########     here according to our variables
#########     theta = ((X'X)^-1)X'Y  this is implemented below

theta=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
print("optimized variables = ",theta)

#function value at input of given X
# y = c + mx
fn=theta[0] + theta[1]*X[:,1]
#print("fn=",fn)

#################   Plotting   ##################
plt.scatter(X[:,1], Y)
plt.plot(X[:,1],fn,'r')
plt.xlabel('Years')
plt.ylabel('Debt')
plt.legend(["Observations", "Linear regression model"], loc ="lower right")
plt.show()


#########################################################################
############           Question No 5 Part (b)               #############
#########################################################################

debt_at2020 = theta[0] + theta[1]*2020
print("debt at 2020 = ",debt_at2020)


#########################################################################
############           Question No 5 Part (c)               #############
#########################################################################

debt_at2041 = theta[0] + theta[1]*2041
print("debt at 2041 = ",debt_at2041)

## the model is giving pretty accurate results  :)