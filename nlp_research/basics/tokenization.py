import re
from collections import defaultdict

import numpy as np
import pandas as pd


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


class TfidfVectorizer:
    def __init__(self):
        self._bow_vectorizer = BagOfWordsVectorizer()
        self.idf_ = None
        self.vocabulary_ = None
        self.word_to_index_ = None

    def fit(self, corpus):
        """
        Builds the vocabulary and calculates IDF values.

        Args:
            corpus (list of str): A list of text documents.
        """
        # Use our BoW vectorizer to get the vocabulary and term counts
        term_counts_matrix = self._bow_vectorizer.fit_transform(corpus)

        # Inherit the vocabulary from the BoW instance
        self.vocabulary_ = self._bow_vectorizer.vocabulary_
        self.word_to_index_ = self._bow_vectorizer.word_to_index_

        n_documents = len(corpus)

        # Calculate document frequency (how many docs contain each word)
        # A non-zero count means the word is in the document
        doc_freq = np.sum(term_counts_matrix > 0, axis=0)

        # Calculate IDF
        # Using log(N / (df + 1)) + 1 for stability and smoothing
        self.idf_ = np.log((n_documents) / (doc_freq + 1)) + 1
        return self

    def transform(self, corpus):
        """
        Transforms a corpus of documents into a TF-IDF matrix.

        Args:
            corpus (list of str): A list of text documents.
        """
        if self.vocabulary_ is None:
            raise RuntimeError('Vectorizer has not been fitted yet. Call fit() first.')

        # Get the raw term counts using the fitted BoW vectorizer
        term_counts_matrix = self._bow_vectorizer.transform(corpus)

        # Calculate Term Frequency (TF)
        # Sum of counts for each document (row)
        doc_term_counts = term_counts_matrix.sum(axis=1)

        # To avoid division by zero for empty documents
        doc_term_counts[doc_term_counts == 0] = 1

        # Divide each element by the total count in its row
        tf_matrix = term_counts_matrix / doc_term_counts[:, np.newaxis]

        # Calculate TF-IDF
        tfidf_matrix = tf_matrix * self.idf_

        return tfidf_matrix

    def fit_transform(self, corpus):
        """Convenience method to fit and then transform."""
        self.fit(corpus)
        return self.transform(corpus)


def main():
    # Sample corpus
    corpus = [
        'The cat sat on the mat.',
        'The dog ate my homework.',
        'The cat and the dog are friends.',
        'Is it a cat or a dog?',
    ]

    # --- Using Bag-of-Words ---
    print('--- Bag-of-Words ---')
    bow_vec = BagOfWordsVectorizer()
    bow_matrix = bow_vec.fit_transform(corpus)

    # Display the result in a readable format
    bow_df = pd.DataFrame(bow_matrix, columns=bow_vec.vocabulary_)
    print('Vocabulary:', bow_vec.vocabulary_)
    print('BoW Matrix:\n', bow_df)

    # --- Using TF-IDF ---
    print('\n\n--- TF-IDF ---')
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(corpus)

    # Display the result in a readable format
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_vec.vocabulary_)
    print('Vocabulary:', tfidf_vec.vocabulary_)
    print('TF-IDF Matrix:\n', tfidf_df.round(2))  # Round for better readability


if __name__ == '__main__':
    main()
