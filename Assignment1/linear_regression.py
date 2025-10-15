import numpy as np

class LinearRegression():
	
	def __init__(self):
		# NOTE: Feel free to add any hyperparameters 
		# (with defaults) as you see fit
		self.learning_rate = 0.001 # Learning rate for gradient descent
		self.num_iterations = 0 # Initialize
		self.max_iterations = 1000000 # Max iterations for gradient descent
		self.beta0 = 0.0  # Intercept
		self.beta1 = 0.0  # Slope
		self.mse = float('inf') # Set first MSE value to infinity
		self.tolerance = 0.000000001 # Tolerance for stopping criteria, i.e when MSE change is less than this value, stop training
		
	def fit(self, X, y):
		"""
		Estimates parameters for the classifier
		
		Args:
			X (array<m,n>): a matrix of floats with
				m rows (#samples) and n columns (#features)
			y (array<m>): a vector of floats
		"""

		X = np.array(X).flatten()
		y = np.array(y).flatten()

		while True:
			test1 = self.mse
			test2 = (1/len(X)) * sum((y - (self.beta0 + self.beta1 * X))**2)

			# Check if change is low enough to stop
			if(abs(self.mse - ((1/len(X)) * sum((y - (self.beta0 + self.beta1 * X))**2))) < self.tolerance or self.num_iterations > self.max_iterations):
				break

			# Update MSE to new value (for next iteration)
			self.mse = (1/len(X)) * sum((y - (self.beta0 + self.beta1 * X))**2)

			# Gradient descent update
			
			temp_beta0 = self.beta0 - self.learning_rate * (1/len(X)) * sum((self.beta0 + self.beta1 * X - y))
			temp_beta1 = self.beta1 - self.learning_rate * (1/len(X)) * sum((self.beta0 + self.beta1 * X - y) * X)

			self.beta0 = temp_beta0
			self.beta1 = temp_beta1

			self.num_iterations += 1
	
	def predict(self, X):
		"""
		Generates predictions
		
		Note: should be called after .fit()
		
		Args:
			X (array<m,n>): a matrix of floats with 
				m rows (#samples) and n columns (#features)
			
		Returns:
			A length m array of floats
		"""
		return self.beta0 + self.beta1 * X




