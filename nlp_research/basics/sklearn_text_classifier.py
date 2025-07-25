import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


class TextClassifier:
    """
    A simple text classifier that categorizes text into:
    - "Technical issue"
    - "Billing issue"
    - "Other"
    """

    def __init__(self):
        """Initialize the classifier with a pipeline."""
        self.pipeline = Pipeline(
            [
                (
                    'tfidf',
                    TfidfVectorizer(
                        lowercase=True, stop_words='english', max_features=1000, ngram_range=(1, 2)
                    ),
                ),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        self.is_trained = False

    def _create_training_data(self):
        """Create synthetic training data for the three categories."""
        training_data = {
            'text': [
                # Technical issues
                'the app is not working',
                'website is down',
                'system error occurred',
                'application crashed',
                'login not working',
                "page won't load",
                'software bug',
                'technical problem',
                'system malfunction',
                'app keeps freezing',
                'website is slow',
                'connection error',
                'server error',
                'technical issue',
                'system failure',
                # Billing issues
                'payment declined',
                'billing problem',
                'charge dispute',
                'double charge',
                'payment failed',
                'billing error',
                'subscription issue',
                'payment method',
                'invoice problem',
                'chargeback request',
                'billing question',
                'payment issue',
                'charge error',
                'billing dispute',
                'payment problem',
                # Other
                'how to change password',
                'account settings',
                'user guide',
                'help needed',
                'general question',
                'information request',
                'how to use',
                'account setup',
                'profile update',
                'contact support',
                'general inquiry',
                'help with account',
                'how to access',
                'account information',
                'support request',
            ],
            'category': (['Technical issue'] * 15 + ['Billing issue'] * 15 + ['Other'] * 15),
        }
        return pd.DataFrame(training_data)

    def train(self, X=None, y=None):
        """
        Train the classifier with synthetic data or provided data.

        Args:
            X: Optional training texts
            y: Optional training labels
        """
        if X is None or y is None:
            # Use synthetic data
            df = self._create_training_data()
            X = df['text']
            y = df['category']

        # Train the pipeline
        self.pipeline.fit(X, y)
        self.is_trained = True

        # Print training accuracy
        y_pred = self.pipeline.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f'Training accuracy: {accuracy:.2f}')

    def predict(self, text):
        """
        Predict the category of the given text.

        Args:
            text (str): Input text to classify

        Returns:
            str: Predicted category
        """
        if not self.is_trained:
            raise ValueError('Classifier must be trained before making predictions')

        # Handle edge cases
        if not text or not text.strip():
            return 'Other'

        # Clean the text
        text = self._preprocess_text(text)

        # Make prediction
        prediction = self.pipeline.predict([text])[0]
        return prediction

    def _preprocess_text(self, text):
        """Clean and preprocess the input text."""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def predict_proba(self, text):
        """
        Get prediction probabilities for all categories.

        Args:
            text (str): Input text to classify

        Returns:
            dict: Dictionary with category probabilities
        """
        if not self.is_trained:
            raise ValueError('Classifier must be trained before making predictions')

        text = self._preprocess_text(text)
        probabilities = self.pipeline.predict_proba([text])[0]
        categories = self.pipeline.classes_

        return dict(zip(categories, probabilities, strict=False))


def run_tests():
    """Run test cases to verify the classifier works correctly."""
    print('=== Text Classifier Test Results ===\n')

    # Initialize and train classifier
    classifier = TextClassifier()
    classifier.train()

    # Test cases
    test_cases = [
        ('My payment was declined', 'Billing issue'),
        ('The app keeps crashing when I open it', 'Technical issue'),
        ('How do I change my password?', 'Other'),
        ('I was charged twice for the same service', 'Billing issue'),
        ('The website is not loading properly', 'Technical issue'),
        ('Where can I find my account settings?', 'Other'),
        ('', 'Other'),  # Edge case
        ('I need help with my subscription renewal', 'Billing issue'),
    ]

    correct = 0
    total = len(test_cases)

    for text, expected in test_cases:
        try:
            prediction = classifier.predict(text)
            confidence = max(classifier.predict_proba(text).values())
            status = '✓' if prediction == expected else '✗'
            print(f"{status} Input: '{text}'")
            print(
                f'   Expected: {expected}, Predicted: {prediction} (confidence: {confidence:.2f})'
            )
            print()

            if prediction == expected:
                correct += 1

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"✗ Error processing '{text}': {e}")
            print()

    print(f'Test Results: {correct}/{total} correct ({correct / total * 100:.1f}%)')


def main():
    # Example usage
    print('=== Example Usage ===')
    classifier = TextClassifier()
    classifier.train()

    example_texts = [
        'The payment system is not working',
        'I was overcharged for my subscription',
        'How do I reset my password?',
    ]

    for text in example_texts:
        prediction = classifier.predict(text)
        probabilities = classifier.predict_proba(text)
        print(f"Text: '{text}'")
        print(f'Prediction: {prediction}')
        print(f'Probabilities: {probabilities}')
        print()


if __name__ == '__main__':
    main()
