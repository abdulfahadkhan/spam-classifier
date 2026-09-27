# Spam Classifier

A text classifier that distinguishes spam from ham (non-spam) SMS messages using TF-IDF features and classic ML models.

## Dataset
[SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,574 labeled SMS messages (86.6% ham, 13.4% spam).

## Approach
1. Preprocessing: lowercasing, punctuation removal, stopword removal
2. Feature extraction: TF-IDF with unigrams + bigrams
3. Models compared: Naive Bayes, Logistic Regression (with `class_weight='balanced'`), Linear SVM

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 95.1% | 1.00 | 0.63 | 0.77 |
| **Logistic Regression (balanced)** | **97.7%** | **0.98** | **0.85** | **0.91** |
| Linear SVM | 97.1% | 1.00 | 0.79 | 0.88 |

Best model: Logistic Regression with `class_weight='balanced'` and bigram features, balancing precision and recall on the imbalanced dataset.

## Usage
\`\`\`bash
pip install -r requirements.txt
python3 spam_classifier_v2.py
\`\`\`
