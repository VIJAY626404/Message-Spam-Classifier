# SMS Spam Classifier
Here are some examples of spam messages classified by the model:

### Model Interface 
![Before Input](https://github.com/VIJAY626404/Message-Spam-Classifier/raw/main/output/spam1.png)
### Output Interface
![After Input ](https://github.com/VIJAY626404/Message-Spam-Classifier/raw/main/output/spam2.png)


## Overview
The SMS Spam Classifier is a machine learning project designed to classify text messages as either "spam" or "ham" (not spam). The model utilizes natural language processing (NLP) techniques and various algorithms to effectively identify spam messages based on their content.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)


## Features
- Classifies SMS messages as spam or ham.
- Uses natural language processing techniques for text analysis.
- Provides insights on the model's performance with accuracy metrics.

## Technologies Used
- **Python**: Programming language for implementing the model.
- **Streamlit**: Framework for creating the web application.
- **NLTK**: Library for natural language processing.
- **Scikit-learn**: Machine learning library for model training and evaluation.
- **Pandas**: Library for data manipulation and analysis.

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sms-spam-classification
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
3. Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```
## Usage
1. Run the application:
   ```bash
     streamlit run app.py
   ```
## Data
The model is trained on the SMS Spam Collection Dataset which contains a set of SMS messages labeled as "spam" or "ham".
## Model Training
The classification model is built using the following steps:

1. Data Preprocessing:
- Text normalization (lowercasing, removing punctuation).
- Tokenization.
- Vectorization (using TF-IDF).
  
2. Model Selection:
- Trained using [mention any algorithms you used, e.g., Logistic Regression, Naive Bayes].
  
3. Model Evaluation:
Accuracy, precision, recall, and F1-score are calculated to evaluate the model's performance.
