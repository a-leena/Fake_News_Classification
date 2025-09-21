# Fake News Classification
This project explores the problem of classifying news articles as *fake* or *real* using Natural Language Processing (NLP). The approach is progressive, starting with exploratory analysis, preprocessing, and baseline machine learning models, and advancing to stronger neural network-based models, including CNNs, LSTMs, and Transformers.

## Dataset
* Source: [Hugging Face](https://huggingface.co/datasets/Reyansh4/Fake-News-Classification)
* Size: 20800 rows
* Features: `title`, `author`, `text`
* Target: `label` (0:Real, 1:Fake)

## Workflow
### 1. Data Preprocessing
* Handle missing values and duplicates
* Clean text (normalize quotes, expand contractions, strip punctuation and whitespace, normalize case)
* Tokenization and lemmatization using **NLTK**

### 2. Exploratory Analysis & Feature Extraction
* Class distribution
* Most frequent unigrams, bigrams, and trigrams
* Word clouds for Fake vs Real news articles
* Stylometric features:
    * **Lexical:** average word length, vocabulary richness
    * **Syntactic:** average sentence length, punctuation ratios, POS ratios
    * **Readability:** Flesch Readability Ease Score

### 3. Baseline Models
* Feature extraction: **TF-IDF**
* Models tested:
    * Logistic Regression
    * Naive Bayes
    * Linear SVC (Support Vector Classifier)
* Experiments conducted with and without stylometric features
* **Result:** Linear SVC with stylometric features performed best, achieving **96.5%** accuracy.
> ✅ This serves as the **baseline** for comparison with neural network models.

### 4. Word Embeddings
* A **Word2Vec** model is trained on the training set to obtain **task-specific embeddings**
* Simple CNN model is used to **tune hyperparameters** (embedding dimension, context window size, maximum input length) efficiently
* **Result:** 
    * Embedding dimension = 500
    * Context window size = 5
    * Maximum input length = 588 (90th percentile length)
    * Simple CNN Accuracy = **96.5%**
> These embeddings and inputs are later used in more complex CNN, LSTM, and Transformer models.

### 5. Convolutional Neural Networks (CNNs)
* CNNs capture **local phrase-level patterns** but cannot model long-term sequential dependencies like LSTMS or Transformers
* A **1D convolutional kernel of size n** acts as an **n-gram detector**
* CNNs of different depths are tuned to find the best **hyperparameters** ( filters, kernel sizes, dense layer units, dropout rate, and learning rate)
* **Results:**
    * CNN models of 3 different depths (1 convolutional layer, 2 layers, and 3 layers) were tuned. CNN model with a single convolutional layer performed the best.
    * Best tuned model achieved **96.3%** accuracy.
> 🔍 Classical ML models (linear SVC) performed competitively with deep learning models on this dataset, even narrowly outperforming them. \
> 🔍 Interestingly, the simple exploratory CNN model (with arbitrarily chosen hyperparameters) used in (4) also slightly outperformed more complex tuned CNNs. Thus, for this dataset, added complexity did not translate to improved performance. \
> These tuned CNNs serve as the **neural network benchmark** for comparison with sequential models.

### ⏭️ Next Steps
* LSTM and BiLSTM models for capturing sequential dependencies
* Pretrained embeddings (e.g., GloVe, FastText) for transfer learning
* Transformer-based architectures (e.g., BERT) for state-of-the-art performance