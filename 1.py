import pandas as pd
import gzip
import json

def load_metadata_products(metadata_path, search_keywords):
    """Load product IDs from metadata based on search keywords."""
    product_ids = set()
    with gzip.open(metadata_path, 'rt', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            title = record.get('title', '').lower()
            description = ' '.join(record.get('description', [])).lower() if isinstance(record.get('description', []), list) else record.get('description', '').lower()
            if any(keyword.lower() in title or keyword.lower() in description for keyword in search_keywords):
                product_ids.add(record['asin'])
    return product_ids

def preprocess_review_text(text):
    """Placeholder for text preprocessing function."""
    # Implement text preprocessing here
    return text

def load_reviews_for_products(reviews_path, product_ids):
    """Load and preprocess reviews for selected products."""
    relevant_reviews = []
    with gzip.open(reviews_path, 'rt', encoding='utf-8') as file:
        for line in file:
            review = json.loads(line)
            if review['asin'] in product_ids:
                review['processed_reviewText'] = preprocess_review_text(review.get('reviewText', ''))
                relevant_reviews.append(review)
    return pd.DataFrame(relevant_reviews)

def main(reviews_path, metadata_path, search_keywords):
    """Main function to process and filter reviews based on metadata search."""
    product_ids = load_metadata_products(metadata_path, search_keywords)
    filtered_reviews_df = load_reviews_for_products(reviews_path, product_ids)
    
    # Optional: Perform additional processing on filtered_reviews_df here
    
    return filtered_reviews_df

if __name__ == "__main__":
    search_keywords = ['Headphones']
    reviews_path = 'C:\\Users\\ommeh\\Downloads\\ir-3\\Electronics_5.json.gz'
    metadata_path = 'C:\\Users\\ommeh\\Downloads\\ir-3\\meta_Electronics.json.gz'
    
    processed_data = main(reviews_path, metadata_path, search_keywords)

    # Save the head of processed_data to a text file for inspection
    with open('ans1.txt', 'w', encoding='utf-8') as file:
        file.write(processed_data.head().to_string())

    # Save processed data to a CSV file
    processed_data.to_csv('processed_data.csv', index=False)
