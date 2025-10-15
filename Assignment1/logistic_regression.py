import numpy as np

class LogisticRegression():
	
	def __init__(self):
		self.learning_rate = 0.01
		self.num_iterations = 0
		self.max_iterations = 1000000
		self.celoss = float('inf')
		self.tolerance = 0.000000001
		self.beta0 = 0.0   # Bias term
		self.betas = None  # Feature weights (will be initialized in fit)

	def fit(self, X, y):
		"""
		Estimates parameters for the classifier
		Args:
			X (array<m,n>): m rows (#samples), n cols (#features)
			y (array<m>): labels (0 or 1)
		"""
		X = np.array(X)
		if X.ndim == 1:
			X = X.reshape(-1, 1) # Reshape if single feature
		m, n = X.shape # m = #samples, n = #features

		y = np.array(y).flatten()

		# initialize betas
		self.betas = np.zeros(n)

		while True:
			# Linear part
			z = self.beta0 + np.dot(X, self.betas)

			# Predict probabilities
			y_pred = self.sigmoid(z)

			# Cross Entropy Loss
			ce_loss_new = -(1/m) * np.sum(y*np.log(y_pred+1e-15) + (1-y)*np.log(1-y_pred+1e-15))

			# Check for stopping condition
			if abs(self.celoss - ce_loss_new) < self.tolerance or self.num_iterations > self.max_iterations:
				break

			self.celoss = ce_loss_new

			# Gradient descent updates
			grad_beta0 = np.sum(y_pred - y) / m
			grad_betas = np.dot(X.T, (y_pred - y)) / m

			self.beta0 -= self.learning_rate * grad_beta0
			self.betas -= self.learning_rate * grad_betas

			self.num_iterations += 1
	
	def predict(self, X):
		"""
		Generates probability predictions for X
		"""
		X = np.array(X)
		if X.ndim == 1:
			X = X.reshape(-1, 1) # Reshape if single feature
		
		z = self.beta0 + np.dot(X, self.betas)
		return self.sigmoid(z)
	
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))




