from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from utils import load_json_lines, preprocess_text, encode_labels


def main():
    train_texts, train_labels = load_json_lines('datasets/train_set.json')
    test_texts, test_labels = load_json_lines('datasets/test_set.json')

    print("Preprocessing train data...")
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    print("Preprocessing test data...")
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        max_features=20000,
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        max_df=0.8
    )

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.7
    )
    
    vectorizer = FeatureUnion([
    ("char", char_vectorizer),
    ("word", word_vectorizer)
    ])
    print("Fitting TF-IDF vectorizer and transforming train data...")
    X_train = vectorizer.fit_transform(train_texts)
    print("Transforming test data...")
    X_test = vectorizer.transform(test_texts)

    clf = LinearSVC(class_weight='balanced', max_iter=20000, random_state=42)

    print("Training Linear SVM model...")
    clf.fit(X_train, train_labels)
    print("Training completed.")

    preds = clf.predict(X_test)

    # Evaluation
    print(f"Test accuracy: {accuracy_score(test_labels, preds):.4f}")
    print("Classification report:")
    print(classification_report(test_labels, preds))

if __name__ == "__main__":
    main()
