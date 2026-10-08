# Fake News Detection Using NLP & Machine Learning

An end-to-end **Fake News Detection System** that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **FAKE** or **GENUINE**.

The project processes news text, extracts meaningful features using **TF-IDF**, trains and compares multiple machine-learning classification algorithms, selects the best-performing model, and provides an interactive interface for testing new news content.

---

## 📌 Project Overview

Fake news can spread rapidly through online platforms, making manual verification difficult at scale.

This project aims to provide an automated **text-based fake news classification system**.

The system learns linguistic patterns from labeled news articles and predicts whether new news content is more similar to the **FAKE** or **GENUINE** class.

### Important Note

This project is a **machine-learning classifier**, not a real-time fact-checking system.

It does not independently verify news against live websites, government sources, or fact-checking databases. The prediction is based on patterns learned from the training dataset.

---

## 🎯 Objectives

The main objectives of this project are to:

- Build an automated fake news classification system.
- Apply NLP techniques to real-world news text.
- Clean and normalize a large news dataset.
- Convert text into numerical features using TF-IDF.
- Train multiple machine-learning classification models.
- Compare models using standard evaluation metrics.
- Perform cross-validation and hyperparameter tuning.
- Analyze incorrectly classified articles.
- Identify informative words/features.
- Save the trained model as a reusable pipeline.
- Provide an interactive interface for testing new news articles.

---

## 🧠 System Workflow

The complete workflow of the project is:

```text
                WELFake Dataset
                       │
                       ▼
                Dataset Loading
                       │
                       ▼
               Dataset Inspection
                       │
                       ▼
             Data Cleaning & Normalization
                       │
                       ▼
             Title + Article Text
                       │
                       ▼
              NLP Preprocessing
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Lowercase    Stop Words    Lemmatization
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Train/Test Split
                       │
                       ▼
                     TF-IDF
                       │
                       ▼
              Numerical Features
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Logistic      Naive Bayes   Linear SVM
     Regression                    Random Forest
          └────────────┼────────────┘
                       ▼
                Model Evaluation
                       │
                       ▼
              Best Model Selection
                       │
                       ▼
             Pipeline Serialization
                       │
                       ▼
              New News Article
                       │
                       ▼
               FAKE / GENUINE
```

---

## 📊 Dataset

The project uses the **WELFake Dataset**.

### Dataset Summary

| Property | Value |
|---|---:|
| Original records | 72,134 |
| Clean records | 63,676 |
| Fake articles | 28,886 |
| Genuine articles | 34,790 |
| Training split | 80% |
| Testing split | 20% |
| Label `0` | FAKE |
| Label `1` | GENUINE |

The dataset contains news titles, article text, and labels.

### Main Dataset Columns

```text
title
text
label
Unnamed: 0
```

The system combines the **title** and **article body** into a single text representation before NLP processing.

---

## 🧹 Data Cleaning

Before training the models, the dataset goes through several cleaning and normalization steps.

The project performs:

- Missing-value handling
- Duplicate removal
- Text validation
- Automatic column identification
- Title and article-text consolidation
- Label normalization
- Empty-text filtering
- Data-quality checks

The labels are standardized as:

```text
0 → FAKE
1 → GENUINE
```

The project also performs a data-flow audit to ensure that the dataset is not artificially truncated during preprocessing.

---

## 📝 NLP Preprocessing

Raw news text cannot be directly given to most traditional machine-learning algorithms.

Therefore, the project applies an NLP preprocessing pipeline.

### Preprocessing Pipeline

```text
Raw News Text
      ↓
Lowercasing
      ↓
URL Removal
      ↓
HTML Removal
      ↓
Special Character Handling
      ↓
Whitespace Normalization
      ↓
Tokenization
      ↓
Stop Word Removal
      ↓
Lemmatization
      ↓
Processed Text
```

### Example

Raw text:

```text
BREAKING!!! The Government has announced a NEW policy.
Visit https://example.com for more information.
```

After preprocessing, unnecessary elements are removed and the text is transformed into a cleaner representation suitable for feature extraction.

---

## 🔎 Exploratory Data Analysis

The project performs Exploratory Data Analysis (EDA) before model training.

The analysis includes:

- Class distribution
- Article-length distribution
- Most frequent words
- Word clouds
- N-gram frequency analysis
- Fake vs. genuine vocabulary analysis

### N-Grams

The project analyzes:

- **Unigrams** — individual words
- **Bigrams** — two-word combinations
- **Trigrams** — three-word combinations

This helps identify common linguistic patterns in the dataset.

---

## 🔢 Feature Extraction — TF-IDF

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical features.

Machine-learning models cannot directly understand raw sentences, so TF-IDF represents the text mathematically.

### Configuration

```text
Maximum Features : 50,000
N-Gram Range     : (1, 2)
```

This means the system considers both:

- Individual words
- Two-word combinations

TF-IDF gives greater importance to terms that are useful for distinguishing documents while reducing the influence of very common terms.

---

## 🤖 Machine Learning Models

Four classification algorithms are trained and compared.

### 1. Logistic Regression

A widely used classification algorithm that learns the relationship between TF-IDF features and the target classes.

### 2. Multinomial Naive Bayes

A probabilistic algorithm commonly used for text classification.

### 3. Linear SVM

A Support Vector Machine classifier that finds a decision boundary capable of separating the two classes.

### 4. Random Forest

An ensemble learning algorithm that combines multiple decision trees to make predictions.

---

## 🏆 Model Performance

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

### Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **Linear SVM** | **97.16%** | **97.16%** | **97.16%** | **97.16%** |
| Logistic Regression | 96.27% | 96.28% | 96.27% | 96.27% |
| Random Forest | 92.90% | 92.98% | 92.90% | 92.87% |
| Multinomial Naive Bayes | 89.54% | 89.64% | 89.54% | 89.55% |

### 🥇 Best Model

**Linear SVM** achieved the best overall performance.

```text
Accuracy  : 97.16%
Precision : 97.16%
Recall    : 97.16%
F1 Score  : 97.16%
```

Therefore, Linear SVM is selected as the **champion model** for the final prediction pipeline.

---

## 📈 Model Evaluation

The project uses multiple evaluation techniques to understand model performance.

### Accuracy

Measures the percentage of total predictions that are correct.

### Precision

Measures how often predictions for a particular class are correct.

### Recall

Measures how many actual examples of a class are successfully identified.

### F1 Score

Provides a balance between precision and recall.

### Confusion Matrix

The project generates confusion matrices to analyze:

- Correct FAKE predictions
- Incorrect FAKE predictions
- Correct GENUINE predictions
- Incorrect GENUINE predictions

---

## 🔄 Cross-Validation

The selected model is further evaluated using **5-fold cross-validation**.

The training data is divided into five parts, and the model is trained and evaluated multiple times using different combinations of these parts.

This helps determine whether the model's performance is consistent across different subsets of the training data.

---

## ⚙️ Hyperparameter Tuning

The project performs hyperparameter tuning for the selected linear classifier.

The `C` parameter is evaluated using:

```text
0.1
0.5
1.0
2.0
```

Grid Search with cross-validation is used to identify a suitable configuration based on weighted F1 score.

---

## 🔬 Feature Importance

The project analyzes informative features from the best-performing model.

This helps identify words and phrases that contribute strongly toward the classification of news as:

```text
FAKE
```

or

```text
GENUINE
```

Feature analysis provides additional insight into what the model has learned from the dataset.

---

## ❌ Error Analysis

The project also examines incorrectly classified test samples.

This helps answer questions such as:

- Which articles are difficult for the model?
- Where does the classifier make mistakes?
- Are some fake articles written similarly to genuine articles?
- Are some genuine articles using language patterns associated with fake news?

Error analysis is useful for understanding the limitations of the model rather than relying only on the overall accuracy.

---

## 💾 Model Serialization

After selecting the best model, the project creates an end-to-end Scikit-learn pipeline containing:

```text
TF-IDF Vectorizer
        +
Champion Classifier
```

The trained pipeline is saved using Joblib:

```text
models/fake_news_detector.pkl
```

This allows the trained model to be loaded later without retraining it from the beginning.

---

## 🖥️ Interactive Prediction

The Jupyter Notebook includes an interactive fake-news prediction interface.

A user can enter:

- A news headline
- A complete news article
- News text for testing

The system then performs:

```text
User Input
    ↓
NLP Preprocessing
    ↓
TF-IDF Transformation
    ↓
Linear SVM
    ↓
Prediction
```

The final prediction is:

```text
FAKE
```

or:

```text
GENUINE
```

along with a model classification confidence-style score.

---

## 📁 Project Structure

```text
Fake_News_Detection_System/
│
├── data/
│   └── WELFake_Dataset.csv
│
├── models/
│   └── fake_news_detector.pkl
│
├── outputs/
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── confusion_matrices.png
│   │   ├── feature_importance.png
│   │   ├── length_distributions.png
│   │   ├── model_comparison.png
│   │   ├── ngram_frequency.png
│   │   ├── top_words.png
│   │   └── wordclouds.png
│   │
│   └── results/
│       └── model_comparison.csv
│
├── fake_news_detection.ipynb
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation |
| NLTK | Natural Language Processing |
| TF-IDF | Text feature extraction |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| WordCloud | Word-frequency visualization |
| Joblib | Model serialization |
| Jupyter Notebook | Development and experimentation |
| IPyWidgets | Interactive prediction interface |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Fake_News_Detection_System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
fake_news_detection.ipynb
```

Run the notebook cells from top to bottom.

The notebook will:

1. Load the dataset.
2. Inspect the data.
3. Clean and normalize the dataset.
4. Perform NLP preprocessing.
5. Generate EDA visualizations.
6. Create TF-IDF features.
7. Train four machine-learning models.
8. Compare their performance.
9. Generate confusion matrices.
10. Perform cross-validation.
11. Tune model hyperparameters.
12. Perform error analysis.
13. Save the final model.
14. Provide interactive predictions.

---

## 📋 Requirements

The project dependencies are listed in:

```text
requirements.txt
```

Main dependencies include:

```text
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
wordcloud
ipywidgets
joblib
jupyter
nbformat
```

NLTK resources are automatically downloaded by the notebook when required.

---

## 🧪 Example Prediction

Example workflow:

```text
Input:
News article text
        ↓
Preprocessing
        ↓
TF-IDF
        ↓
Linear SVM
        ↓
Output:
GENUINE
```

or:

```text
Input:
News article text
        ↓
Preprocessing
        ↓
TF-IDF
        ↓
Linear SVM
        ↓
Output:
FAKE
```

## 🚀 Future Enhancements

Possible improvements include:

- Integration with real-time news sources.
- Integration with trusted fact-checking APIs.
- Use of transformer models such as BERT or RoBERTa.
- Multilingual fake-news detection.
- Explainable AI for individual predictions.
- Source credibility analysis.
- Claim-level fact checking.
- Web-based deployment.
- REST API for model predictions.
- Continuous model retraining using new verified data.
- Improved handling of short headlines and social-media posts.

---

## 📊 Key Results

The final system achieved the following performance on the test dataset:

```text
                    Linear SVM

Accuracy   : 97.16%
Precision  : 97.16%
Recall     : 97.16%
F1 Score   : 97.16%
```

The final trained pipeline is stored at:

```text
models/fake_news_detector.pkl
```

---

## 🔑 Key Takeaway

The project demonstrates how traditional NLP and machine-learning techniques can be combined to build an effective fake-news classification system.

The overall process is:

```text
News Text
   ↓
Data Cleaning
   ↓
NLP Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Linear SVM
   ↓
FAKE / GENUINE
```

---

## 👨‍💻 Project

**Fake News Detection Using NLP & Machine Learning**

Built as an academic machine-learning project demonstrating:

- Natural Language Processing
- Text Classification
- Feature Engineering
- Machine Learning
- Model Evaluation
- Error Analysis
- Model Serialization
- Interactive Prediction
