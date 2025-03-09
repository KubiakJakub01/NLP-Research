import numpy as np


def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    N = len(corpus)
    tfidf = []
    idf = []
    for q in query:
        df = sum(1 if q in document else 0 for document in corpus)
        idf.append(np.log((N + 1) / (df + 1)) + 1)

    for document in corpus:
        document_len = len(document)
        if document_len == 0:
            tfidf.append([0.0] * len(query))
            continue
        tf = [document.count(q) / document_len for q in query]

        tfidf.append([round(_tf * _idf, 4) for _tf, _idf in zip(tf, idf, strict=False)])

    return tfidf
