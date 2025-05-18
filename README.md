# Sentiment-to-hate-analysis
From Sentiment to Hate: Sentence-Level Emotion Classification and Hate Speech Detection in Machine Learning

## 📊 Dataset
The provided sentiment_dataset.csv contains tweets categorized into three classes:
0 – Negative sentiment
1 – Neutral sentiment
2 – Hate speech

The class labels must be encoded as {0, 1, 2} because the CrossEntropyLoss function used in training expects integer class indices starting from 0 up to num_labels - 1.

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

## 📊 Hyperparameter Grid
The model is fine-tuned using a small grid:\

`param_grid = list(ParameterGrid({
    'learning_rate': [2e-5, 3e-5],
    'per_device_train_batch_size': [8, 16]
}))`

The evaluation batch size is set equal to the training batch size for consistency.

## 📐 Statistical Comparison
After collecting results over 5 folds, Friedman tests are run for each metric to determine if different hyperparameter configurations yield statistically different results.



# Convolutional Neural Network
## Project Overview
- Data loading
- Data processing: cleaning, tokenization, creating dictionary of 10 000 most common tokens
- Training CNN
- Collection of metrics: F1, accuracy, precision, recall, confusion matrix
- Hyperparameters (learning rate, batch size, number of epochs) tuning
- Friedman statistical tests to determine if hyperparameter tuning leads to statistically significant differences

## Model
```python
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
TextCNN                                  [32, 3]                   --
├─Embedding: 1-1                         [32, 100, 100]            1,000,200
├─Conv1d: 1-2                            [32, 100, 98]             30,100
├─ReLU: 1-3                              [32, 100, 98]             --
├─AdaptiveMaxPool1d: 1-4                 [32, 100, 1]              --
├─Linear: 1-5                            [32, 3]                   303
==========================================================================================
Total params: 1,030,603
```

## 📊 Hyperparameter tuning
- learning rate = 1e-03, 1e-04
- batch size = 16, 32, 64
- epochs = 10, 15

## 📐 Statistical Comparison
After collecting results over 5 folds, Friedman tests are run for each metric to determine if different hyperparameter configurations yield statistically different results.

## Requirements 
`pip install pytorch pandas datasets scikit-learn scikit-posthocs seaborn matplotlib`




# LSTM Sentiment Classification with Cross-Validation, Hyperparameter Tuning, and Statistical Analysis

## Project Overview
- Data loading and preprocessing using `Tokenizer` and `pad_sequences`
- LSTM-based architecture using `TensorFlow/Keras`
- Grid search over `lstm_units`, `dropout_rate`, `batch_size`
- 5-fold stratified cross-validation for model evaluation
- Metric collection: Accuracy, Precision, Recall, F1-score (per fold and config)
- Confusion matrix visualization for best configuration
- Statistical comparison using Friedman test with Nemenyi post-hoc analysis

## 📊 Hyperparameter Grid
The model is fine-tuned using a small grid:\

`param_grid = list(ParameterGrid({
    'lstm_units': [32, 64],
    'dropout_rate': [0.3, 0.5],
}))
`
## Manually specified (non-default) hyperparameters of the model architecture:

- input_dim=10000 - sice of the vocabulary (based on the number of unique words)
- output_dim=200 - length of embeding vector, value 200 is commonly use for the model. It was not tunned because of it low impact on model and to reduce computatino time
- Dense(64, activation='relu') – a dense layer with ReLU activation (it allowes model to lern more complex representatinos)
- Dense(3, activation='softmax') - defining how many class we have
- Nr of epochs - 10

## Requirements
pip install tensorflow scikit-learn scikit-posthocs seaborn matplotlib

## 📐 Statistical Comparison
After collecting results over 5 folds, Friedman tests are run for each metric to determine if different hyperparameter configurations yield statistically different results.

# SVM Sentiment Classification with TF-IDF, Cross-Validation, and Metric Analysis

## Project Overview
- Feature extraction using `TfidfVectorizer` with English stopwords removal
- Training a Support Vector Machine (SVM) classifier with a linear kernel
- 5-fold stratified cross-validation for performance evaluation
- Metric collection: Accuracy, Precision, Recall, F1-score for each fold
- Final confusion matrix calculated across all folds
- Saving per-fold metrics to CSV file for further analysis or statistical comparison

## Requirements
pip install scikit-learn pandas matplotlib seaborn





