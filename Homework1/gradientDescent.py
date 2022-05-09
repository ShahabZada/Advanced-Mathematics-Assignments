#from Student_debt import X, Y
from computeCost import *
import numpy as np

def  gradientDescent(X, Y, theta, alpha, num_iters):
    # theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    # taking num_iters gradient steps with learning rate alpha

    size = len(X)
    print("size=",size)
    J_history = np.array([])

    for i in range(num_iters):   
        
        #(1/(2*size)) * sum( ((c + m * X) - Y) ** 2)
        t1 = np.sum(Y - (theta[0] + theta[1] * X) ) # Un-Vectorized
        t2 = np.sum((Y - (theta[0] + theta[1] * X)) * X) # Un-Vectorized
        #print("t1 = ",t1)
        theta[0] = theta[0] - (alpha/size) * (t1)
        theta[1] = theta[1] - (alpha/size) * (t2)
        
        # Save the cost J in every iteration    
        print("cost",computeCost(X, Y, theta))
        np.append(J_history,computeCost(X, Y, theta))

    return [theta, J_history] 

