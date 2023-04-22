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
