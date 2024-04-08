import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import importlib

# Importing necessary components from '5.py'
module_5 = importlib.import_module("5")

# Setup environment
def setup_environment():
    print("Setting up environment...")

def load_and_preprocess_dataset(file_path):
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(file_path)
    df['rating_class'] = df['overall'].apply(lambda x: 'Good' if x > 3 else ('Average' if x == 3 else 'Bad'))
    
    # Fill NaN values with an empty string
    df['processed_reviewText'].fillna('', inplace=True)
    
    return df

# Split dataset
def split_dataset(df, features_col='processed_reviewText', target_col='rating_class'):
    print("Splitting dataset...")
    X = df[features_col]
    y = LabelEncoder().fit_transform(df[target_col])
    return train_test_split(X, y, test_size=0.25, random_state=42)

# Define machine learning models
def define_models():
    print("Defining models...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    return {
        "Logistic Regression": Pipeline([('Vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')), ('Classifier', LogisticRegression(max_iter=500))]),
        "MultinomialNB": Pipeline([('Vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')), ('Classifier', MultinomialNB())]),
        "LinearSVC": Pipeline([('Vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')), ('Classifier', LinearSVC(dual=False))]),
        "Decision Tree": Pipeline([('Vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')), ('Classifier', DecisionTreeClassifier(max_depth=10))]),
        "KNN with SVD": Pipeline([('Vectorizer', TfidfVectorizer(max_features=5000, stop_words='english')), ('SVD', svd), ('Classifier', KNeighborsClassifier(n_neighbors=5))])
    }

# Evaluate models and write reports to file
def evaluate_and_write_reports(models, X_train, X_test, y_train, y_test, output_file='model_reports.txt'):
    print("Evaluating models and writing reports...")
    with open(output_file, 'w') as file:
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=['Bad', 'Average', 'Good'])
            print(report)
            file.write(f"Model: {name}\n{report}\n\n")

# Main execution function
def main():
    setup_environment()
    file_path = 'processed_data.csv'
    df = load_and_preprocess_dataset(file_path)
    X_train, X_test, y_train, y_test = split_dataset(df)
    models = define_models()
    evaluate_and_write_reports(models, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
