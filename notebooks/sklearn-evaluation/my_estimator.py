import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

class my_lr_estimator():
	
	def __init__(self, l2_reg=5.):
		self.l2_reg=l2_reg
		self.needless_attr = 2
	
	def fit(self, X, y):
		X, y = check_X_y(X,y)
		num_features = X.shape[1]
		X = np.insert(X,num_features, 1., axis=1)
		num_features = X.shape[1]
		self.w_coef_ = np.linalg.pinv(X.T.dot(X)+self.l2_reg*np.eye(num_features)).dot(X.T).dot(y)
		return self
	
	def predict(self, X):
		check_is_fitted(self, ["w_coef_"])
		X = check_array(X)
		num_features = X.shape[1]
		X = np.insert(X,num_features, 1., axis=1)
		return X.dot(self.w_coef_)
		
	def score(self, X, y):
		prediction = self.predict(X)
		num_samples = y.shape[0]
		sq_diff = (y-prediction)**2
		sum_sq_diff = sq_diff
		while len(sum_sq_diff.shape)>1:
			sum_sq_diff = sq_diff.sum(axis=1)
		return -sum_sq_diff.mean()
	
	def get_params(self, deep=True):
		return {"l2_reg": self.l2_reg}
		
	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			self.setattr(parameter, value)
		return self
	
	def setattr(self, parameter, value):
		setattr(self, parameter, value)