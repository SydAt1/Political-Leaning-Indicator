import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logger import setup_logger

# Set up logging
logger = setup_logger('nlp_vectorize')

def vectorize_text(input_path='data/processed/cleaned_history.csv', output_path='data/processed/tfidf_features.pkl'):
    """Vectorize cleaned text using TF-IDF and save the result."""
    try:
        # Load cleaned data
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {input_path} with {len(df)} records")

        # Combine cleaned_title and cleaned_url for vectorization
        text_data = df['cleaned_title'].fillna('') + ' ' + df['cleaned_url'].fillna('')

        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(text_data)

        # Convert to DataFrame or save as pickle
        feature_names = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

        # Save the vectorized features
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tfidf_df.to_pickle(output_path)
        logger.info(f"Saved TF-IDF features to {output_path} with shape {tfidf_matrix.shape}")

        return tfidf_matrix, vectorizer

    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        raise

if __name__ == "__main__":
    vectorize_text()