import json
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def load_json_lines(filepath):
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            combined_text = f"{entry.get('headline', '')} {entry.get('short_description', '')}"
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

    # Define vectorizers
    char_vect = TfidfVectorizer(analyzer='char_wb')
    word_vect = TfidfVectorizer(analyzer='word')

    # Pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('char', char_vect),
            ('word', word_vect)
        ])),
        ('clf', MultinomialNB())
    ])

    # Parameter grid
    param_grid = {
        'features__char__ngram_range': [(3, 5), (2, 4)],
        'features__char__min_df': [1, 3],
        'features__char__max_df': [0.7, 0.9],
        'features__char__sublinear_tf': [True],

        'features__word__ngram_range': [(1, 2), (1, 3)],
        'features__word__min_df': [1, 3],
        'features__word__max_df': [0.7, 0.9],
        'features__word__sublinear_tf': [True],

        'clf__alpha': [0.01, 0.05, 0.1, 0.5, 1.0],
    }

    # Grid Search
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    print("Running grid search...")
    grid.fit(train_texts, train_labels)

    print(f"\n✅ Best Accuracy: {grid.best_score_:.4f}")
    print("🔧 Best Parameters:")
    for param, val in grid.best_params_.items():
        print(f"  {param}: {val}")

    # Final test evaluation
    print("\n📊 Evaluating on test set...")
    preds = grid.best_estimator_.predict(test_texts)
    print(f"Test Accuracy: {accuracy_score(test_labels, preds):.4f}")
    print("Classification Report:")
    print(classification_report(test_labels, preds))

if __name__ == "__main__":
    main()
