import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import importlib
from bs4 import BeautifulSoup
import warnings

# Importing from '1.py'
module_1 = importlib.import_module("1")

# Initialize NLTK resources
def setup_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Define text preprocessing functions
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_accented_chars(text):
    return unidecode.unidecode(text)

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def to_lowercase(text):
    return text.lower()

def tokenize(text):
    return text.split()

def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word not in stop_words]

def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

# Preprocess text
def preprocess_text_v2(text):
    if not isinstance(text, str) or not text.strip():
        return ""  # Early return for non-string or empty input

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = clean_html(text)
    
    text = remove_accented_chars(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    words = tokenize(text)
    words = remove_stopwords(words)
    meaningful_words = lemmatize_words(words)
    return " ".join(meaningful_words)

# Process DataFrame
def process_dataframe(file_path):
    df = pd.read_csv(file_path)
    df['processed_reviewText'] = df['processed_reviewText'].apply(preprocess_text_v2)
    return df

# Save processed data
def save_preprocessed_data(df, file_name='fully_preprocessed_data_v2.csv'):
    df.to_csv(file_name, index=False)

# Main function
def execute_preprocessing_pipeline():
    setup_nltk_resources()
    file_path = 'processed_data.csv'  # Specify your actual file path
    processed_df = process_dataframe(file_path)
    save_preprocessed_data(processed_df)
    
    # Writing the output to a text file
    with open('ans5.txt', 'w', encoding='utf-8') as f:
        # Convert DataFrame head to a string and write to the file
        f.write(processed_df.head().to_string())

if __name__ == "__main__":
    execute_preprocessing_pipeline()
