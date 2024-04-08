import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import unidecode
import nltk
from bs4 import BeautifulSoup
import warnings
import importlib

# Importing from '5.py'
module_5 = importlib.import_module("5")

# Setup NLTK resources
def setup_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Preprocess text function encapsulation
def preprocess_text(text):
    """Preprocess the text by cleaning and lemmatizing it."""
    return module_5.preprocess_text_v2(text)

# Data loading and preprocessing
def load_and_preprocess(file_path):
    """Load the data and apply preprocessing."""
    df = pd.read_csv(file_path)
    df['processed_reviewText'] = df['processed_reviewText'].apply(preprocess_text)
    return df

# Save data to CSV
def save_data(df, file_name):
    """Save the DataFrame to a CSV file."""
    df.to_csv(file_name, index=False)

# Generate and save word cloud
def save_word_cloud(df, title, file_name):
    """Generate a word cloud from the DataFrame and save it to a file."""
    text = " ".join(review for review in df['processed_reviewText'])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.savefig(file_name)
    plt.close()

# Main function to execute pipeline
def main(file_path, output_csv, wordcloud_file, output_txt):
    setup_nltk()
    df = load_and_preprocess(file_path)
    save_data(df, output_csv)
    
    # Writing output to a text file
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(df.head().to_string())
    
    # Save a word cloud image
    save_word_cloud(df, "Processed Data Word Cloud", wordcloud_file)

if __name__ == "__main__":
    file_path = 'processed_data.csv'  # Input CSV file
    output_csv = 'fully_preprocessed_data_v2.csv'  # Output CSV file
    wordcloud_file = 'data_word_cloud.png'  # Word cloud image file
    output_txt = 'ans6.txt'  # Text file for DataFrame head
    
    main(file_path, output_csv, wordcloud_file, output_txt)
