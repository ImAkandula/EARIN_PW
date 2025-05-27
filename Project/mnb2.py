import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(lemmatizer.lemmatize(token) for token in tokens)

def load_dataset(filepath):
    texts, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            combined = f"{entry['headline']} {entry['short_description']}"
            texts.append(combined)
            labels.append(entry['category'])
    return pd.DataFrame({'text': texts, 'label': labels})

def main():
    train_path = 'datasets/2k.json'  # Path to your training JSONL
    test_path = 'datasets/400.json'    # Path to your test JSONL

    # Load datasets
    print("Loading training data...")
    train_df = load_dataset(train_path)
    print(f"Training samples: {train_df.shape[0]}")
    
    print("Loading test data...")
    test_df = load_dataset(test_path)
    print(f"Test samples: {test_df.shape[0]}")

    # Lemmatize texts
    print("Lemmatizing training data...")
    train_df['text'] = train_df['text'].apply(lemmatize_text)
    
    print("Lemmatizing test data...")
    test_df['text'] = test_df['text'].apply(lemmatize_text)

    # Oversample training data to handle imbalance
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(train_df[['text']], train_df['label'])
    print(f"Training samples after oversampling: {X_resampled.shape[0]}")

    # Define pipeline: TF-IDF + Multinomial Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=20000,
            
            ngram_range=(1, 2),
            max_df=0.7,
            min_df=10
        )),
        ('nb', MultinomialNB(alpha=0.1))
    ])

    # Train the model
    print("Training model...")
    pipeline.fit(X_resampled['text'], y_resampled)

    # Predict on test set
    print("Predicting on test data...")
    preds = pipeline.predict(test_df['text'])

    # Evaluate
    print("\nAccuracy:", accuracy_score(test_df['label'], preds))
   
if __name__ == "__main__":
    main()
