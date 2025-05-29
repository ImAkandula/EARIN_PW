import warnings
import time
from tqdm import tqdm
from collections import Counter
from utils import load_json_lines, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("Loading dataset...")
    train_texts, train_labels = load_json_lines('datasets/train_set.json')
    test_texts, test_labels = load_json_lines('datasets/test_set.json')

    print(f"Training samples: {len(train_texts)} | Testing samples: {len(test_texts)}")
    print(f"Sample category counts: {Counter(train_labels).most_common(5)}")

    print("Preprocessing train data...")
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    print("Preprocessing test data...")
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

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

    vectorizer = FeatureUnion([
        ("char", char_vectorizer),
        ("word", word_vectorizer)
    ])

    print("Fitting TF-IDF vectorizer on training data...")
    X_train = vectorizer.fit_transform(train_texts)
    print("Transforming test data...")
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(
    max_iter=20000,
    class_weight='balanced',
    solver='liblinear',
    C=1,
    random_state=42
)

    print("Training the model...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    print(f"Model training completed in {end_time - start_time:.2f} seconds.\n")

    print("Evaluating model on test data...")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}\n")


if __name__ == "__main__":
    main()
