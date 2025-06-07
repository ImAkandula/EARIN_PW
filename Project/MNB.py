import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_json_lines, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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



def train_and_evaluate(X_train, y_train, X_test, y_test, label_names, out_prefix):
    clf = MultinomialNB(alpha=0.05)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=label_names)

    print(f"[{out_prefix}] Test Accuracy: {acc:.4f}")
    print(report)

    with open(f"{out_prefix}_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title(f"Confusion Matrix - MNB ({out_prefix})")
    plt.tight_layout()
    plt.savefig(f"confusion-matrix-mnb-{out_prefix}.png")
    plt.close()


def main():
    print("Loading data...")
    train_texts, train_labels = load_json_lines("datasets/train_set.json")
    test_texts, test_labels = load_json_lines("datasets/test_set.json")

    print("Preprocessing...")
    train_texts = [preprocess_text(t) for t in tqdm(train_texts)]
    test_texts = [preprocess_text(t) for t in tqdm(test_texts)]

    # TF-IDF zgodnie z Twoimi parametrami
    vectorizer = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            max_features=20000,
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=2,
            max_df=0.8
        )),
        ("word", TfidfVectorizer(
            analyzer="word",
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.7
        ))
    ])

    print("Vectorizing...")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print("\n--- MNB without label grouping ---")
    le_full = LabelEncoder()
    y_train_full = le_full.fit_transform(train_labels)
    y_test_full = le_full.transform(test_labels)
    train_and_evaluate(X_train, y_train_full, X_test, y_test_full, le_full.classes_, "1")

    print("\n--- MNB with grouped labels ---")
    train_labels_grouped = [simplify_label(l) for l in train_labels]
    test_labels_grouped = [simplify_label(l) for l in test_labels]

    le_grouped = LabelEncoder()
    y_train_grouped = le_grouped.fit_transform(train_labels_grouped)
    y_test_grouped = le_grouped.transform(test_labels_grouped)
    train_and_evaluate(X_train, y_train_grouped, X_test, y_test_grouped, le_grouped.classes_, "2")


if __name__ == "__main__":
    main()
