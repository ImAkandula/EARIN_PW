import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import json
from tqdm import tqdm
from utils import load_json_lines, preprocess_text
from sklearn.preprocessing import LabelEncoder

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

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

def generate_confusion_matrices(model_dir='saved_bert_model', test_file='datasets/test_set.json', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # Load label mappings
    with open(f'{model_dir}/label2id.json', 'r') as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # Load test data
    print("Loading test data...")
    test_texts, test_labels = load_json_lines(test_file)
    test_labels_encoded = [label2id[label] for label in test_labels]

    # Tokenize data
    print("Tokenizing test data...")
    test_dataset = NewsDataset(test_texts, test_labels_encoded, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inference
    model.eval()
    preds = []
    true_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Map integer ids to original labels
    pred_labels = [id2label[p] for p in preds]
    true_labels_text = [id2label[t] for t in true_labels]

    # Before Grouping
    print("\n[Before Grouping]")
    labels_full = np.unique(true_labels_text + pred_labels)
    le_full = LabelEncoder()
    y_true = le_full.fit_transform(true_labels_text)
    y_pred = le_full.transform(pred_labels)
    
    acc_before = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc_before:.4f}")
    report_before = classification_report(y_true, y_pred, target_names=le_full.classes_)
    print(report_before)

    with open('bert_before_grouping_report.txt', 'w') as f:
        f.write(f"Accuracy: {acc_before:.4f}\n")
        f.write(report_before)

    cm_before = confusion_matrix(y_true, y_pred)
    disp_before = ConfusionMatrixDisplay(confusion_matrix=cm_before, display_labels=le_full.classes_)
    fig, ax = plt.subplots(figsize=(20, 20))
    disp_before.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix - BERT (Before Grouping)")
    plt.tight_layout()
    plt.savefig("bert_confusion_matrix_before_grouping.png")
    plt.close()

    # After Grouping
    print("\n[After Grouping]")
    pred_labels_grouped = [simplify_label(lbl) for lbl in pred_labels]
    true_labels_grouped = [simplify_label(lbl) for lbl in true_labels_text]

    labels_grouped = np.unique(true_labels_grouped + pred_labels_grouped)
    le_grouped = LabelEncoder()
    y_true_grouped = le_grouped.fit_transform(true_labels_grouped)
    y_pred_grouped = le_grouped.transform(pred_labels_grouped)
    
    acc_after = accuracy_score(y_true_grouped, y_pred_grouped)
    print(f"Accuracy: {acc_after:.4f}")
    report_after = classification_report(y_true_grouped, y_pred_grouped, target_names=le_grouped.classes_)
    print(report_after)

    with open('bert_after_grouping_report.txt', 'w') as f:
        f.write(f"Accuracy: {acc_after:.4f}\n")
        f.write(report_after)

    cm_after = confusion_matrix(y_true_grouped, y_pred_grouped)
    disp_after = ConfusionMatrixDisplay(confusion_matrix=cm_after, display_labels=le_grouped.classes_)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp_after.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix - BERT (After Grouping)")
    plt.tight_layout()
    plt.savefig("bert_confusion_matrix_after_grouping.png")
    plt.close()

if __name__ == "__main__":
    generate_confusion_matrices()
