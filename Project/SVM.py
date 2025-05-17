import json
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

def load_json_lines(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry['headline'])
            labels.append(entry['category'])
    return texts, labels

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

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    print("Fitting TF-IDF vectorizer and transforming train data...")
    X_train = vectorizer.fit_transform(train_texts)
    print("Transforming test data...")
    X_test = vectorizer.transform(test_texts)

    # Initialize Linear SVM classifier
    clf = LinearSVC(class_weight='balanced', max_iter=100000, random_state=42)

    print("Training Linear SVM model...")
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
