import json
import re
import warnings
import time
from tqdm import tqdm
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load JSON Lines file (one JSON object per line)
def load_json_lines(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                headline = entry.get('headline', '')
                description = entry.get('short_description', '')
                label = entry.get('category')
                if not label:
                    continue  # Skip if label missing
                combined_text = f"{headline} {description}"
                texts.append(combined_text)
                labels.append(label)
            except json.JSONDecodeError:
                continue  # Skip corrupted lines
    return texts, labels

# Preprocess text: lowercase and remove non-alphabetic characters
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("🔄 Loading dataset...")
    train_texts, train_labels = load_json_lines('datasets/train_set.json')
    test_texts, test_labels = load_json_lines('datasets/test_set.json')

    print(f"✅ Training samples: {len(train_texts)} | Testing samples: {len(test_texts)}")
    print(f"📊 Sample category counts: {Counter(train_labels).most_common(5)}")

    # Preprocess text data
    print("🧹 Preprocessing train data...")
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    print("🧹 Preprocessing test data...")
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    # Encode string labels into integers
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

    # TF-IDF vectorizers
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 7),
        max_features=20000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    )

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    )

    # Combine vectorizers
    vectorizer = FeatureUnion([
        ("char", char_vectorizer),
        ("word", word_vectorizer)
    ])

    print("⚙️ Fitting TF-IDF vectorizer on training data...")
    X_train = vectorizer.fit_transform(train_texts)
    print("⚙️ Transforming test data...")
    X_test = vectorizer.transform(test_texts)

    # Logistic Regression classifier with multithreaded solver
    clf = LogisticRegression(
        solver='saga',
        penalty='l2',  # Efficient and supports multi-threading
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        random_state=42,
        multi_class='multinomial',
        n_jobs=-1,
        verbose=1  # Logs progress in terminal
    )

    print("🚀 Training the model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    print(f"✅ Model training completed in {end_time - start_time:.2f} seconds.\n")

    # Predict and evaluate
    print("🔎 Evaluating model on test data...")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 Test Accuracy: {acc:.4f}\n")


if __name__ == "__main__":
    main()
