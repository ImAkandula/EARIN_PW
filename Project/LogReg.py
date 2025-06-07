import warnings
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from utils import load_json_lines, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore", category=UserWarning)

# --- category/class grouping ---
WORLD_POLITICS_GROUP = {
    "POLITICS", "WORLD NEWS", "U.S. NEWS", "WORLDPOST", "THE WORLDPOST"
}
ARTS_GROUP = {
    "ARTS", "ARTS & CULTURE", "CULTURE & ARTS"
}
PARENTING_GROUP = {
    "PARENTING", "PARENTS"
}

def simplify_label(label):
    if label in WORLD_POLITICS_GROUP:
        return "WORLD_POLITICS"
    elif label in ARTS_GROUP:
        return "ARTS_GROUP"
    elif label in PARENTING_GROUP:
        return "PARENTING_GROUP"
    return label

def main():
    print("Loading dataset...")
    train_texts, train_labels = load_json_lines('datasets/train_set.json')
    test_texts, test_labels = load_json_lines('datasets/test_set.json')

    print("Simplifying labels (grouping)...")
    train_labels = [simplify_label(label) for label in train_labels]
    test_labels = [simplify_label(label) for label in test_labels]

    print("Label distribution after grouping:")
    print(Counter(train_labels).most_common())

    print("Preprocessing text...")
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

    print("Final class labels used:", list(le.classes_))

    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 7), max_features=20000, sublinear_tf=True, min_df=3, max_df=0.9)
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=20000, sublinear_tf=True, min_df=3, max_df=0.9)
    vectorizer = FeatureUnion([("char", char_vectorizer), ("word", word_vectorizer)])

    print("Vectorizing...")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        solver='liblinear',
        C=1,
        random_state=42
    )

    print("Training Logistic Regression...")
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.\n")

    print("Evaluating model on test data...")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix - Logistic Regression with Grouped Labels")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
