from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from utils.logger import setup_logger
import pickle
import os

# Set up logging
logger = setup_logger('nlp_vectorizer')

def vectorize_text(df, max_features=500, output_path='data/processed/tfidf_features.pkl'):
    """
    Vectorize the cleaned text data using TF-IDF and save the matrix.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 'cleaned_title' column.
        max_features (int): Maximum number of features for TF-IDF (default: 500).
        output_path (str): Path to save the TF-IDF matrix (default: 'data/processed/tfidf_features.pkl').
    
    Returns:
        tuple: (tfidf_matrix, vectorizer) - TF-IDF matrix and fitted vectorizer.
    """
    try:
        if 'cleaned_title' not in df.columns:
            raise ValueError("DataFrame must contain 'cleaned_title' column")
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        
        # Vectorize the cleaned titles
        tfidf_matrix = vectorizer.fit_transform(df['cleaned_title'].fillna(''))
        
        # Save TF-IDF matrix
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        logger.info(f"Vectorized {tfidf_matrix.shape[0]} documents with {tfidf_matrix.shape[1]} features and saved to {output_path}")
        
        return tfidf_matrix, vectorizer
    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage for testing
    df = pd.read_csv('data/processed/cleaned_history.csv')
    tfidf_matrix, vectorizer = vectorize_text(df)