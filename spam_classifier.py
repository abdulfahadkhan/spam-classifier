import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
print(df.head())
print(df['label'].value_counts())

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_message'] = df['message'].apply(preprocess)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

def evaluate(model, name):
    preds = model.predict(X_test)
    print("\n--- " + name + " ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=['ham', 'spam']))

evaluate(nb_model, "Naive Bayes")
evaluate(lr_model, "Logistic Regression")

def predict_message(msg, model=lr_model):
    clean = preprocess(msg)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]
    return "SPAM" if result == 1 else "HAM"

print("\nTest prediction:", predict_message("WIN a free iPhone now, click this link!!!"))
print("Test prediction:", predict_message("Hey, are we still meeting for lunch tomorrow?"))
