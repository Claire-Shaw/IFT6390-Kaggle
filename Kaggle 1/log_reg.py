import numpy as np


class MultiClassLogisticRegression:
    def __init__(self, reg=0.1, stepsize=0.01, batch_size=5000, n_steps=500, normalize=True):
        self.reg = reg
        self.stepsize = stepsize
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.normalize=normalize
    
    def data_normalization(self, X, train=False):
        if train==True:
            self.means = np.mean(X, axis=0)
        
        X -= self.means

        if train==True:
            self.stds = np.std(X, axis=0)
        X = X / self.stds 

        return X

    def _predict(self, X):
        return np.matmul(-X, self.weights)
        
    def _softmax(self, X):
        scaled_X = X - X.max(axis=1, keepdims=True)
        exp_X = np.exp(scaled_X)
        softmax_preds = np.divide(exp_X, exp_X.sum(axis=1, keepdims=True))
        return softmax_preds

    def _loss(self, X, y):
        preds = self._predict(X)
        norm_factor = 1/X.shape[0]
        loss = norm_factor * (np.trace(X @ self.weights @ y.T) + np.sum(np.log(np.sum(np.exp(preds), axis=1))))
        return loss

    def _gradient(self, X, y):
        preds = self._predict(X)
        softmax_preds = self._softmax(preds)
        norm_fact = 1 / X.shape[0]
        gd = norm_fact * np.matmul(X.T, (y - softmax_preds)) + 2 * self.reg * self.weights
        return gd
    
    def train(self, train_x, train_y):
        self.classes = np.unique(train_y).astype(int)
        
        if self.normalize:
            X = self.data_normalization(train_x, train=True)
        else:
            X = train_x
        
        # One-hot encode the y values
        y = np.eye(len(self.classes))[train_y.astype(int).reshape(-1)]

        self.weights = np.zeros((X.shape[1], y.shape[1]))
        
        losses = []

        np.random.seed(42)
        for epoch in range(self.n_steps):

            if self.batch_size:
                idx = np.random.choice(X.shape[0], self.batch_size)
                batch_x, batch_y = X[idx], y[idx]
            else:
                batch_x, batch_y = X, y
            
            loss = self._loss(batch_x, batch_y)
            self.weights -= self.stepsize * self._gradient(batch_x, batch_y)

            losses.append(loss)
            print(f'epoch {epoch}:      Loss: {loss}')
        
        return losses

    def predict(self, X):
        if self.normalize:
            X = self.data_normalization(X)

        preds = self._predict(X)
        softmax_preds = self._softmax(preds)
        class_labels = np.argmax(softmax_preds, axis=1)
        
        return class_labels
