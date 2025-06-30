import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        """
        Constructor for the Logistic Regression classifier.

        Args:
            learning_rate (float): Step size for Gradient Descent.
            n_iters (int): Number of iterations for Gradient Descent.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """The sigmoid 'squashing' function."""
        return 1 / (1 + np.exp(-z))

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
            # Calculate the linear model: z = w*x + b
            linear_model = np.dot(X, self.weights) + self.bias

            # Apply the sigmoid function to get probabilities
            y_predicted = self._sigmoid(linear_model)  # This is ŷ

            # 3. Calculate gradients
            # Derivative of cost w.r.t weights (dw)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # Derivative of cost w.r.t bias (db)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict class labels for new data.

        Args:
            X (np.array): Test data features.

        Returns:
            list: A list of predicted class labels (0 or 1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)

        # Convert probabilities to class labels (0 or 1) using a 0.5 threshold
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


def main():
    # Load and prepare the data
    data = load_breast_cancer()
    X, y = data.data, data.target  # pylint: disable=no-member

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Use our Logistic Regression Classifier ---

    # 1. Instantiate the classifier
    # These hyperparameters might need tuning for different datasets
    clf = LogisticRegression(learning_rate=0.01, n_iters=1000)

    # 2. Train the classifier
    clf.fit(X_train, y_train)

    # 3. Make predictions on the test data
    predictions = clf.predict(X_test)

    # 4. Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print('Custom Logistic Regression Classifier')
    print(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
