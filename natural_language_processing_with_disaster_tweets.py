import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train["text"] = train["text"].apply(preprocess_text)
test["text"] = test["text"].apply(preprocess_text)
train["keyword"] = train["keyword"].fillna("").apply(preprocess_text)
test["keyword"] = test["keyword"].fillna("").apply(preprocess_text)

train["combined_text"] = train["keyword"] + " " + train["text"]
test["combined_text"] = test["keyword"] + " " + test["text"]

X = train["combined_text"]
y = train["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_val_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
print("Accuracy:", accuracy)
print("F1 score:", f1)

X_test = test["combined_text"]
X_test_tfidf = vectorizer.transform(X_test)
test_preds = model.predict(X_test_tfidf)

submission = pd.DataFrame({"id": test["id"], "target": test_preds})
submission.to_csv("submission.csv", index=False)
