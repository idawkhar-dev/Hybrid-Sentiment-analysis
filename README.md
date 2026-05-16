# Hybrid-Sentiment-analysis

A hybrid deep learning framework combining CNN and LSTM for accurate sentiment detection on Twitter data.

Problem Statement
Social media platforms like Twitter generate vast amounts of unstructured text data characterized by slang, sarcasm, and ambiguity. Traditional sentiment analysis methods whether lexicon-based or standalone machine learning models struggle to handle these linguistic nuances accurately. Lexicon-based approaches lack contextual understanding, while deep learning models are often computationally expensive and inaccessible to smaller stakeholders.

Project Objective
To develop a hybrid sentiment analysis model that integrates the strengths of both lexicon-based and machine learning approaches, enabling:
1. Accurate classification of tweets into Positive, Negative, and Neutral sentiments
2. Robust handling of sarcasm, slang, and context-dependent language
3. Improved performance without excessive computational overhead

Method Used
Datasets:
Kaggle Twitter Dataset — 31,962 labelled tweets (positive/negative/neutral), randomly sampled for balanced distribution
Real-Time Twitter Dataset — 600 tweets extracted via the Twitter API, focused on the 2024 US Presidential Elections

Preprocessing:
Noise removal, null value handling, and special character cleaning
Sentiment labelling using TextBlob (lexicon-based model)
Tokenization and sequence padding using TensorFlow's Tokenizer and pad_sequences


