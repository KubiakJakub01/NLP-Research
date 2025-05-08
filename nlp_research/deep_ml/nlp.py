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


def osa(source: str, target: str) -> int:
    """
    Compute the Optimal String Alignment (OSA) distance between two strings.

    :param source: The source string
    :param target: The target string
    :return: The OSA distance between the two strings
    """
    len_s = len(source)
    len_t = len(target)

    d = [[0] * (len_t + 1) for _ in range(len_s + 1)]

    for i in range(len_s + 1):
        d[i][0] = i

    for j in range(len_t + 1):
        d[0][j] = j

    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1

            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if (
                i > 1
                and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)

    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if (
                i > 1
                and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)

    return d[len_s][len_t]
