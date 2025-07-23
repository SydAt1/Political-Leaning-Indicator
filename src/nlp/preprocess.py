import pandas as pd
import spacy
import re
import os
from utils.logger import setup_logger

# Load SpaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # Disable unused components for efficiency

# Set up logging
logger = setup_logger('nlp_preprocess')

def clean_text(text):
    """Clean text by removing URLs, punctuation, and stopwords, and applying lemmatization."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http[s]?://\S+', '', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.text.strip()]
    return ' '.join(tokens)

def preprocess_history(input_path, output_path='data/processed/cleaned_history.csv'):
    """Preprocess browser history CSV and save cleaned data.
    
    Args:
        input_path (str): Path to the raw history CSV file.
        output_path (str): Path to save the cleaned history CSV (default: 'data/processed/cleaned_history.csv').
    
    Returns:
        pandas.DataFrame: Processed DataFrame with cleaned text.
    """
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {input_path} with {len(df)} records")

        df['cleaned_title'] = df['title'].apply(clean_text)
        df['cleaned_url'] = df['url'].apply(clean_text)
        df = df[(df['cleaned_title'] != '') | (df['cleaned_url'] != '')]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned history to {output_path} with {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    input_path = 'data/raw/user1_history.csv'  # Single path for testing
    preprocess_history(input_path)