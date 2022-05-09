# Importing libraries

#from tkinter import W
import numpy as np
import csv
import matplotlib.pyplot as plt

# Linear Regression

class LinearRegression() :
	
	def __init__( self, learning_rate, iterations ) :
		
		self.learning_rate = learning_rate
		
		self.iterations = iterations
		
	# Function for model training
			
	def fit( self, X, Y ) :
		
		# no_of_training_examples, no_of_features
		
		self.m = X.shape[0]
		#print(self.m)
		# weight initialization
		
		self.W = 0
		
		self.b = 0
		
		self.X = X
		
		self.Y = Y
		
		
		# gradient descent learning
				
		for i in range( self.iterations ) :
			#print(self.W)  #debug
			self.update_weights()
			
		return self
	
	# Helper function to update weights in gradient descent
	
	def update_weights( self ) :
			
		Y_pred = self.predict( self.X )
		
		# calculate gradients
	
		dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) / self.m
		#print(dW)  #debug
		db = - 2 * np.sum( self.Y - Y_pred ) / self.m
		
		# update weights
	
		self.W = self.W - self.learning_rate * dW
	
		self.b = self.b - self.learning_rate * db
		
		return self
	
	# Hypothetical function h( x )
	
	def predict( self, X ) :
		print((X*( self.W ) + self.b).shape)
		return np.multiply(X, self.W ) + self.b
	

# driver code

def main() :
	
	# Importing dataset
	file = open('student_debt.csv')
	csvreader = csv.reader(file)
	#header = next(csvreader)
	#print(header)
	#rows = []
	years = []
	debt = []
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
	print(X)
	Y = np.asfarray(debt)

	# Model training
	
	model = LinearRegression( iterations = 1000, learning_rate = 0.0001 )

	model.fit( X, Y )
	
	# Prediction on test set

	#Y_pred = model.predict( X_test )
	
	#print( "Predicted values ", np.round( Y_pred[:3], 2 ) )
	
	#print( "Real values	 ", Y_test[:3] )
	
	print( "Trained W	 ", model.W )
	
	print( "Trained b	 ", model.b )
	
	# Visualization on test set
	"""
	plt.scatter( X_test, Y_test, color = 'blue' )
	
	plt.plot( X_test, Y_pred, color = 'orange' )
	
	plt.title( 'Salary vs Experience' )
	
	plt.xlabel( 'Years of Experience' )
	
	plt.ylabel( 'Salary' )
	
	plt.show()
	"""
if __name__ == "__main__" :
	
	main()
