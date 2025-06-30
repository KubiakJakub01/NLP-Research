import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Constructor for the Linear Regression model.

        Args:
            learning_rate (float): Step size for Gradient Descent.
            n_iters (int): Number of iterations for Gradient Descent.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Args:
            X (np.array): Training data features (n_samples, n_features).
            y (np.array): Training data labels (n_samples,).
        """
        n_samples, n_features = X.shape

        # 1. Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent loop
        for _ in range(self.n_iters):
            # Calculate predictions: y_hat = w*x + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # 3. Calculate gradients
            # Derivative of cost w.r.t weights (dw)
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))

            # Derivative of cost w.r.t bias (db)
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # 4. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict continuous values for new data.

        Args:
            X (np.array): Test data features.

        Returns:
            np.array: An array of predicted values.
        """
        return np.dot(X, self.weights) + self.bias


def main():
    # Generate some sample data
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)  # pylint: disable=unbalanced-tuple-unpacking

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Use our Linear Regression Model ---

    # 1. Instantiate the model
    # A slightly higher learning rate can work for Linear Regression
    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)

    # 2. Train the model
    regressor.fit(X_train, y_train)

    # 3. Make predictions on the test data
    predictions = regressor.predict(X_test)

    # 4. Evaluate the model using Mean Squared Error
    def mse(y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)

    mse_value = mse(y_test, predictions)
    print('Custom Linear Regression Model')
    print(f'Mean Squared Error: {mse_value:.2f}')
    print(f'Learned Weights: w = {regressor.weights[0]:.2f}')
    print(f'Learned Bias: b = {regressor.bias:.2f}')


if __name__ == '__main__':
    main()
