import json
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load JSON Lines data
def load_json_lines(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # Combine headline and short_description
            headline = entry.get('headline', '')
            description = entry.get('short_description', '')
            combined_text = f"{headline} {description}"
            texts.append(combined_text)
            labels.append(entry['category'])
    return texts, labels

# Preprocess text: lowercase + remove non-alpha chars
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # Load data
    train_texts, train_labels = load_json_lines('datasets/train_set.json')
    test_texts, test_labels = load_json_lines('datasets/test_set.json')

    # Preprocess texts
    print("Preprocessing train data...")
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    print("Preprocessing test data...")
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    # Hybrid TF-IDF Vectorizer: char + word
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    )

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    )

    vectorizer = FeatureUnion([
        ("char", char_vectorizer),
        ("word", word_vectorizer)
    ])

    print("Fitting vectorizer and transforming train data...")
    X_train = vectorizer.fit_transform(train_texts)
    print("Transforming test data...")
    X_test = vectorizer.transform(test_texts)

    # Initialize Multinomial Naive Bayes classifier
    clf = MultinomialNB(alpha = 0.05)

    print("Training Complement Naive Bayes model...")
    clf.fit(X_train, train_labels)
    print("Training completed.")

    # Predict on test set
    preds = clf.predict(X_test)

    # Evaluation
    print(f"Test accuracy: {accuracy_score(test_labels, preds):.4f}")
    print("Classification report:")
    print(classification_report(test_labels, preds))

if __name__ == "__main__":
    main()
