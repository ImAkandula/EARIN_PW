import json
import re
from sklearn.preprocessing import LabelEncoder

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
                    continue
                combined_text = f"{headline} {description}"
                texts.append(combined_text)
                labels.append(label)
            except json.JSONDecodeError:
                continue
    return texts, labels

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def encode_labels(train_labels, test_labels):
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    train_encoded = encoder.transform(train_labels)
    test_encoded = encoder.transform(test_labels)
    label2id = {label: idx for idx, label in enumerate(encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}
    return train_encoded, test_encoded, label2id, id2label
