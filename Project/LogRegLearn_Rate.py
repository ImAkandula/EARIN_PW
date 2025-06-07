import numpy as np
from utils import load_json_lines, preprocess_text
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Wczytanie i przygotowanie danych (bez upraszczania etykiet)
train_texts, train_labels = load_json_lines("datasets/train_set.json")
test_texts, test_labels = load_json_lines("datasets/test_set.json")

print("Preprocessing texts...")
train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test = le.transform(test_labels)

# TF-IDF vectorizer (jak wcześniej)
vectorizer = FeatureUnion([
    ("char", TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 7),
        max_features=20000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    )),
    ("word", TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    ))
])

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Lista wartości C
c_values = [0.01, 0.1, 0.3, 1, 3, 10]
results = []

print("Training Logistic Regression for various C values (full 41-label set)...")
for c in c_values:
    clf = LogisticRegression(C=c, class_weight='balanced', solver='liblinear', max_iter=3000, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results.append((c, acc))

# Wyświetlenie wyników
print("\nAccuracy for different values of C (no label grouping):")
print("{:<10}{}".format("C", "Accuracy"))
for c, acc in results:
    print("{:<10}{:.4f}".format(c, acc))
