from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def euclidean_distance(p1, p2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        p1 (np.array): The first point.
        p2 (np.array): The second point.

    Returns:
        float: The Euclidean distance.
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))


class KNN:
    def __init__(self, k=3):
        """
        Constructor for the KNN classifier.

        Args:
            k (int): The number of nearest neighbors to use.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        The "training" step for KNN. All it does is memorize the data.

        Args:
            X_train (np.array): Training data features.
            y_train (np.array): Training data labels.
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predicts the labels for the test data.

        Args:
            X_test (np.array): Test data features.

        Returns:
            list: A list of predicted labels for each point in X_test.
        """
        # Loop through each test point and predict its label
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Helper function to predict the label for a single data point.

        Args:
            x (np.array): A single data point (features).

        Returns:
            The predicted label for the single point.
        """
        # 1. Calculate distances from the new point 'x' to all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # 2. Find the K-nearest neighbors
        # Get the indices of the K smallest distances
        k_neighbor_indices = np.argsort(distances)[: self.k]

        # Get the labels of these neighbors
        k_neighbor_labels = [self.y_train[i] for i in k_neighbor_indices]

        # 3. Vote for the most common label
        # Using Counter is an efficient way to do a majority vote
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


def main():
    # Load the dataset
    iris = load_iris()
    X, y = iris.data, iris.target  # pylint: disable=no-member

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Use our KNN Classifier ---

    # 1. Instantiate the classifier
    k_value = 5
    clf = KNN(k=k_value)

    # 2. "Train" the classifier
    clf.fit(X_train, y_train)

    # 3. Make predictions on the test data
    predictions = clf.predict(X_test)

    # 4. Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Custom KNN Classifier with k={k_value}')
    print(f'Predictions: {predictions}')
    print(f'Actual Labels: {y_test}')
    print(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()
