import json
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def load_json_lines(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            headline = entry.get('headline', '')
            description = entry.get('short_description', '')
            combined_text = f"{headline} {description}"
            texts.append(combined_text)
            labels.append(entry['category'])
    return texts, labels

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # Load and preprocess data
    train_texts, train_labels = load_json_lines('datasets/2k.json')
    test_texts, test_labels = load_json_lines('datasets/400.json')
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    # Pipeline setup
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=2000, random_state=42))
    ])

    # Grid search parameters
    param_grid = {
        'tfidf__max_features': [10000, 20000, 30000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 3],
        'tfidf__max_df': [0.5, 0.7],
        'tfidf__sublinear_tf': [True],
        'clf__C': [0.1, 1.0, 5.0],
        'clf__solver': ['liblinear', 'lbfgs'],
        'clf__class_weight': [None, 'balanced'],
        'clf__penalty': ['l2']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    print("Running grid search...")
    grid.fit(train_texts, train_labels)

    print(f"\n✅ Best Accuracy: {grid.best_score_:.4f}")
    print("🔧 Best Parameters:")
    for param, val in grid.best_params_.items():
        print(f"  {param}: {val}")

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    preds = grid.best_estimator_.predict(test_texts)
    print(f"Test Accuracy: {accuracy_score(test_labels, preds):.4f}")
    print(classification_report(test_labels, preds))

if __name__ == "__main__":
    main()
