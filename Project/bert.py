import json
import torch
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm  # Progress bar


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


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item['headline'])
                labels.append(item['category'])
    return texts, labels


def encode_labels(train_labels, test_labels):
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    train_encoded = encoder.transform(train_labels)
    test_encoded = encoder.transform(test_labels)
    label2id = {label: idx for idx, label in enumerate(encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}
    return train_encoded, test_encoded, label2id, id2label


def train_bert(train_file, test_file, model_dir='saved_bert_model', epochs=3, batch_size=16, max_len=128, lr=2e-5):
    print("Loading training data...")
    train_texts, train_labels = load_data(train_file)
    print(f"Training articles: {len(train_texts)}")
    print("Loading test data...")
    test_texts, test_labels = load_data(test_file)
    print(f"Testing articles: {len(test_texts)}")

    train_encoded, test_encoded, label2id, id2label = encode_labels(train_labels, test_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = NewsDataset(train_texts, train_encoded, tokenizer, max_len)
    test_dataset = NewsDataset(test_texts, test_encoded, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("Training on CPU")

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label2id))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = amp.GradScaler()

    epoch_train_losses = []
    epoch_val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for step, batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            avg_loss_so_far = total_loss / (step + 1)
            loop.set_postfix(loss=avg_loss_so_far)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} — Training loss: {avg_loss:.4f}")
        epoch_train_losses.append(avg_loss)

        # Validate on test set after each epoch to track accuracy curve
        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                preds.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(true_labels, preds)
        print(f"Epoch {epoch+1}/{epochs} — Validation accuracy: {acc:.4f}")
        epoch_val_accuracies.append(acc)

    # Save model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(f'{model_dir}/label2id.json', 'w') as f:
        json.dump(label2id, f)
    print(f"Model saved to {model_dir}")

    # Final evaluation & embeddings extraction
    print("Final evaluation on test set...")
    model.eval()
    preds, true_labels = [], []
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            # Get BERT embeddings (last hidden state CLS token)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings_list.append(cls_embeddings)
            labels_list.append(labels.cpu().numpy())

            logits = model.classifier(outputs.pooler_output)
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Test set accuracy: {acc:.4f}")

    embeddings = np.vstack(embeddings_list)
    labels_np = np.hstack(labels_list)

    # Plot training curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), epoch_train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), epoch_val_accuracies, marker='o', color='orange')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    # t-SNE plot
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels_np, cmap='tab20', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE of BERT CLS Embeddings")
    plt.show()

    # PCA plot
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels_np, cmap='tab20', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("PCA of BERT CLS Embeddings")
    plt.show()


def predict(text, model_dir='saved_bert_model'):
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    import json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    with open(f'{model_dir}/label2id.json') as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label_id = torch.argmax(logits, dim=-1).item()

    return id2label[pred_label_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict news category with BERT")
    parser.add_argument('--train', action='store_true', help="Train BERT model")
    parser.add_argument('--predict', type=str, help="Predict category for given headline")
    parser.add_argument('--train_file', type=str, default='datasets/train_set.json', help="Training data JSON file")
    parser.add_argument('--test_file', type=str, default='datasets/test_set.json', help="Test data JSON file")
    args = parser.parse_args()

    if args.train:
        train_bert(args.train_file, args.test_file)
    elif args.predict:
        category = predict(args.predict)
        print(f"Predicted category: {category}")
    else:
        print("Please specify --train to train or --predict 'headline text' to predict.")
