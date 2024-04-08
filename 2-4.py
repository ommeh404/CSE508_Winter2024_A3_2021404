import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import importlib

# Importing from a file named "1.py"
q1 = importlib.import_module("1")

# Data Loading
def load_data(file_path):
    """Load dataset from the given file path."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data Preprocessing
def preprocess_data(df):
    """Preprocess data by removing duplicates and handling missing values."""
    df_cleaned = df.drop_duplicates().dropna(subset=['processed_reviewText'])
    return df_cleaned

# Descriptive Statistics
def descriptive_stats(df):
    """Calculate and print descriptive statistics of the dataset."""
    stats = {
        'total_rows': len(df),
        'num_reviews': df.shape[0],
        'avg_rating': df['overall'].mean(),
        'unique_products': df['asin'].nunique(),
        'good_ratings': (df['overall'] >= 3).sum(),
        'bad_ratings': (df['overall'] < 3).sum(),
    }
    return stats

# Rating Category
def add_rating_category(df):
    """Add a rating category based on the 'overall' column."""
    df['rating_category'] = np.where(df['overall'] >= 3, 'Good', 'Bad')
    return df

# Visualization: Word Cloud
def generate_word_cloud(text, title):
    """Generate and display a word cloud."""
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
    plt.savefig(r'C:\Users\ommeh\Downloads\ir-3\word_cloud_image.png')

    plt.close() 

# Main Analysis Function
def analyze_data(file_path):
    """Main function to run the data analysis."""
    df = load_data(file_path)
    if df is not None:
        df = preprocess_data(df)
        stats = descriptive_stats(df)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        df = add_rating_category(df)
        good_reviews_text = " ".join(df[df['rating_category'] == 'Good']['processed_reviewText'])
        generate_word_cloud(good_reviews_text, "Good Ratings Word Cloud")

if __name__ == "__main__":
    processed_file_path = 'processed_data.csv'
    analyze_data(processed_file_path)
