from transformers import pipeline


def classify_sentences(sentences: list[str], categories: list[str]) -> list[str]:
    """
    Classify each sentence into one of the predefined categories using a zero-shot model.
    Args:
        sentences: List of sentences to classify.
        categories: List of category labels.
    Returns:
        List of predicted categories (one per sentence).
    """

    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    results = []
    for sentence in sentences:
        output = classifier(sentence, categories)
        # Get the category with the highest score
        predicted = output['labels'][0]
        results.append(predicted)
    return results


def main():
    categories = ['Finance', 'Technology', 'Healthcare']
    sentences = [
        'The stock market crashed yesterday.',
        'New advances in AI are transforming the tech industry.',
        'Doctors recommend regular exercise for a healthy heart.',
        'Bitcoin reached a new all-time high.',
        'A new vaccine was approved by the FDA.',
    ]
    predictions = classify_sentences(sentences, categories)
    for sent, pred in zip(sentences, predictions, strict=False):
        print(f'Sentence: {sent}\nPredicted Category: {pred}\n')


if __name__ == '__main__':
    main()
