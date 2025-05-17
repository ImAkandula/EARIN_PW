import json
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load JSON Lines data loader (use if your JSON file has one JSON object per line)
def load_json_lines(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry['headline'])
            labels.append(entry['category'])
    return texts, labels

# Preprocess text: lowercase + remove non-alpha chars
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # Load data (adjust file path & loader according to your file format)
    train_texts, train_labels = load_json_lines('datasets/train_set.json')
    test_texts, test_labels = load_json_lines('datasets/test_set.json')

    # Preprocess texts
    print("Preprocessing train data...")
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    print("Preprocessing test data...")
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    # TF-IDF Vectorizer setup
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    print("Fitting TF-IDF vectorizer and transforming train data...")
    X_train = vectorizer.fit_transform(train_texts)
    print("Transforming test data with TF-IDF vectorizer...")
    X_test = vectorizer.transform(test_texts)

    # Logistic Regression with class weight balanced
    clf = LogisticRegression(max_iter=20000, class_weight='balanced', solver='liblinear', random_state=42)

    # Training progress: scikit-learn doesn't give progress natively,
    # but we can wrap fit in tqdm by splitting manually or just print before/after.
    print("Training Logistic Regression model...")
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
