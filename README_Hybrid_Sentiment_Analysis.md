# Hybrid Sentiment Analysis on Twitter Dataset

> A hybrid deep learning framework combining CNN and LSTM for accurate sentiment detection on Twitter data.

---

## Problem Statement

Social media platforms like Twitter generate vast amounts of unstructured text data characterized by brevity, slang, sarcasm, and ambiguity. Traditional sentiment analysis methods — whether lexicon-based or standalone machine learning models — struggle to handle these linguistic nuances accurately. Lexicon-based approaches lack contextual understanding, while deep learning models are often computationally expensive and inaccessible to smaller stakeholders.

---

## Project Objective

To develop a **hybrid sentiment analysis model** that integrates the strengths of both lexicon-based and machine learning approaches, enabling:
- Accurate classification of tweets into **Positive**, **Negative**, and **Neutral** sentiments
- Robust handling of sarcasm, slang, and context-dependent language
- Improved performance without excessive computational overhead

---

## Method Used

### Datasets
- **Kaggle Twitter Dataset** — 31,962 labelled tweets (positive/negative/neutral), randomly sampled for balanced distribution
- **Real-Time Twitter Dataset** — 600 tweets extracted via the Twitter API, focused on the 2024 US Presidential Elections

### Preprocessing
- Noise removal, null value handling, and special character cleaning
- Sentiment labelling using **TextBlob** (lexicon-based model)
- Tokenization and sequence padding using TensorFlow's `Tokenizer` and `pad_sequences`

### Model Architecture (CNN + LSTM Hybrid)

| Layer | Description |
|---|---|
| Embedding Layer | Converts word indices to dense vectors (batch size: 100) |
| 1D Convolutional Layer | 128 filters for n-gram feature extraction |
| MaxPooling Layer | Reduces dimensionality, retains key features |
| LSTM Layer | Captures sequential dependencies in text |
| Dense + Dropout Layer | Reduces overfitting via random connection dropout |
| Output Layer | Softmax probability distribution over sentiment classes |

### Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC Curves (AUC)

---

## Results

### Kaggle Dataset
| Sentiment | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.86 | 0.90 | 0.88 |
| Neutral | 0.97 | 0.91 | 0.94 |
| Positive | 0.94 | 0.97 | 0.95 |
| **Weighted Avg** | **0.79** | **0.78** | **0.78** |

### Real-Time Twitter Dataset
| Sentiment | Precision | Recall | F1-Score |
|---|---|---|---|
| Negative | 0.94 | 0.70 | 0.80 |
| Neutral | 0.88 | 0.66 | 0.75 |
| Positive | 0.83 | 0.98 | 0.90 |
| **Weighted Avg** | **0.85** | **0.82** | **0.82** |

- ROC curves for both datasets showed AUC > 0.5 across all classes, confirming reliable model discrimination
- Training and validation loss steadily decreased across epochs, indicating good model convergence
- Word cloud visualizations confirmed meaningful sentiment-word associations across all three classes

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange) ![TextBlob](https://img.shields.io/badge/TextBlob-NLP-green)

`Python` · `TensorFlow/Keras` · `TextBlob` · `NumPy` · `Pandas` · `Matplotlib` · `Tweepy` · `scikit-learn`

---

## Authors

Samiksha Singh · Siddharth Apte · Ishwari Dawkhar
