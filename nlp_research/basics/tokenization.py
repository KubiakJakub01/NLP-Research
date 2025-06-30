import re
from collections import defaultdict

import numpy as np


class BagOfWordsVectorizer:
    def __init__(self):
        self.vocabulary_ = None  # The sorted list of unique words
        self.word_to_index_ = None  # A mapping from word to its index

    def _tokenize(self, text):
        """A simple tokenizer: lowercase and split by non-alphanumeric chars."""
        # Convert to lowercase and split by any character that is not a letter or number
        return re.split(r'\W+', text.lower())

    def fit(self, corpus):
        """
        Builds the vocabulary from a corpus of documents.

        Args:
            corpus (list of str): A list of text documents.

        Returns:
            self: The fitted vectorizer instance.
        """
        all_tokens = set()
        for doc in corpus:
            tokens = self._tokenize(doc)
            for token in tokens:
                if token:  # Ignore empty strings that might result from splitting
                    all_tokens.add(token)

        self.vocabulary_ = sorted(list(all_tokens))
        self.word_to_index_ = {word: i for i, word in enumerate(self.vocabulary_)}
        return self

    def transform(self, corpus):
        """
        Transforms a corpus of documents into a BoW matrix.

        Args:
            corpus (list of str): A list of text documents.

        Returns:
            np.array: The document-term matrix (n_documents, n_vocabulary).
        """
        if self.vocabulary_ is None:
            raise RuntimeError('Vectorizer has not been fitted yet. Call fit() first.')

        # Initialize a zero matrix
        n_documents = len(corpus)
        n_vocab = len(self.vocabulary_)
        bow_matrix = np.zeros((n_documents, n_vocab), dtype=int)

        for i, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            # Use a dictionary for efficient counting within a single document
            word_counts = defaultdict(int)
            for token in tokens:
                word_counts[token] += 1

            # Populate the matrix for the current document
            for word, count in word_counts.items():
                if word in self.word_to_index_:
                    j = self.word_to_index_[word]
                    bow_matrix[i, j] = count

        return bow_matrix

    def fit_transform(self, corpus):
        """Convenience method to fit and then transform."""
        self.fit(corpus)
        return self.transform(corpus)
