# Sentiment-to-hate-analysis
From Sentiment to Hate: Sentence-Level Emotion Classification and Hate Speech Detection in Machine Learning

# 📊 DistilBERT Sentiment Classification with Cross-Validation, Hyperparameter Tuning, and Statistical Analysis
## Project Overview
- Data loading
- Tokenization using Hugging Face's "AutoTokenizer"
- Training and evaluation of DistilBERT using `transformers.Trainer`
- Grid search over learning rate and batch size
- Collection of metrics: F1, accuracy, precision, recall, confusion matrix
- Friedman statistical tests to determine if hyperparameter tuning leads to statistically significant differences
## Requirements 
`pip install transformers datasets scikit-learn scikit-posthocs seaborn matplotlib`
## 📊 Dataset
The provided sentiment_dataset.csv contains tweets categorized into three classes:

0 – Negative sentiment

1 – Neutral sentiment

2 – Hate speech

The class labels must be encoded as {0, 1, 2} because the CrossEntropyLoss function used in training expects integer class indices starting from 0 up to num_labels - 1.

## 📊 Hyperparameter Grid
The model is fine-tuned using a small grid:\

`param_grid = list(ParameterGrid({
    'learning_rate': [2e-5, 3e-5],
    'per_device_train_batch_size': [8, 16]
}))`

The evaluation batch size is set equal to the training batch size for consistency.

## 📐 Statistical Comparison
After collecting results over 5 folds, Friedman tests are run for each metric to determine if different hyperparameter configurations yield statistically different results.
