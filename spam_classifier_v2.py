import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

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

# n-grams added here
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def evaluate(model, name, preds):
    print("\n--- " + name + " ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=['ham', 'spam']))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
evaluate(nb_model, "Naive Bayes (bigrams)", nb_model.predict(X_test))

# Logistic Regression with class_weight balanced
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)
evaluate(lr_model, "Logistic Regression (balanced, bigrams)", lr_model.predict(X_test))

# Logistic Regression with lowered threshold
probs = lr_model.predict_proba(X_test)[:, 1]
preds_adjusted = (probs > 0.3).astype(int)
evaluate(lr_model, "Logistic Regression (threshold=0.3)", preds_adjusted)

# Linear SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
evaluate(svm_model, "Linear SVM (bigrams)", svm_model.predict(X_test))
