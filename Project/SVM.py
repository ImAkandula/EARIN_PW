from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from utils import load_json_lines, preprocess_text


def simplify_label(label):
    if label in {"POLITICS", "WORLD NEWS", "U.S. NEWS", "WORLDPOST", "THE WORLDPOST"}:
        return "WORLD_POLITICS"
    elif label in {"ARTS", "ARTS & CULTURE", "CULTURE & ARTS"}:
        return "ARTS_GROUP"
    elif label in {"PARENTING", "PARENTS"}:
        return "PARENTING_GROUP"
    elif label in {"STYLE", "STYLE & BEAUTY"}:
        return "STYLE_GROUP"
    elif label in {"RELIGION", "QUEER VOICES"}:
        return "IDENTITY_GROUP"
    elif label in {"WEDDINGS", "DIVORCE"}:
        return "LIFESTYLE_EVENTS"
    elif label in {"HEALTHY LIVING", "WELLNESS", "GREEN"}:
        return "HEALTH_GROUP"
    elif label in {"LATINO VOICES", "BLACK VOICES"}:
        return "MINORITY_VOICES"
    elif label in {"SCIENCE", "TECH"}:
        return "STEM_GROUP"
    elif label in {"HOME & LIVING", "TASTE", "FOOD & DRINK"}:
        return "LIFESTYLE_GROUP"
    elif label in {"COLLEGE", "EDUCATION"}:
        return "EDUCATION_GROUP"
    elif label in {"BUSINESS", "IMPACT", "MONEY", "MEDIA"}:
        return "ECONOMY_MEDIA_GROUP"
    else:
        return label


def train_and_evaluate(X_train, y_train, X_test, y_test, labels, out_prefix):
    clf = LinearSVC(class_weight='balanced', max_iter=20000, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=labels)

    print(f"\n[{out_prefix}] Test Accuracy: {acc:.4f}")
    print(report)

    with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, preds, labels=np.arange(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(20, 20))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title(f"Confusion Matrix - Linear SVM ({out_prefix})")
    plt.tight_layout()
    plt.savefig(f"confusion-matrix-svm-{out_prefix}.png")
    plt.close()


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

    # ---- Evaluation 1: Original labels
    print("\n--- Evaluating with Original Labels ---")
    labels_full = np.unique(train_labels)
    label_to_int = {label: idx for idx, label in enumerate(labels_full)}
    y_train = np.array([label_to_int[label] for label in train_labels])
    y_test = np.array([label_to_int[label] for label in test_labels])

    train_and_evaluate(X_train, y_train, X_test, y_test, labels_full, "before-grouping")

    # ---- Evaluation 2: Grouped labels
    print("\n--- Evaluating with Grouped Labels ---")
    train_labels_grouped = [simplify_label(label) for label in train_labels]
    test_labels_grouped = [simplify_label(label) for label in test_labels]
    labels_grouped = np.unique(train_labels_grouped)
    label_to_int_grouped = {label: idx for idx, label in enumerate(labels_grouped)}
    y_train_grouped = np.array([label_to_int_grouped[label] for label in train_labels_grouped])
    y_test_grouped = np.array([label_to_int_grouped[label] for label in test_labels_grouped])

    train_and_evaluate(X_train, y_train_grouped, X_test, y_test_grouped, labels_grouped, "after-grouping")


if __name__ == "__main__":
    main()
