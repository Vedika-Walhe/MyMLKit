class LinearRegressionClosedForm:
    def __init__(self):
        self.theta_best = None

    def fit(self,X,y):
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        self.theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        y_predicted = X.dot(self.theta_best)
        return y_predicted
