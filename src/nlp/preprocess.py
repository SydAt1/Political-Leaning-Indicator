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

def preprocess_history(input_paths, output_path='data/processed/cleaned_history.csv'):
    """Preprocess browser history CSVs and save cleaned data."""
    try:
        dfs = []
        for path in input_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs.append(df)
                logger.info(f"Loaded {path} with {len(df)} records")
            else:
                logger.warning(f"Input file {path} not found")
        if not dfs:
            raise FileNotFoundError("No input files found")
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['cleaned_title'] = combined_df['title'].apply(clean_text)
        combined_df['cleaned_url'] = combined_df['url'].apply(clean_text)
        combined_df = combined_df[
            (combined_df['cleaned_title'] != '') | (combined_df['cleaned_url'] != '')
        ]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned history to {output_path} with {len(combined_df)} records")
        return combined_df
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    input_paths = [
        'data/raw/user1_history.csv',
        'data/raw/user2_history.csv'
    ]
    preprocess_history(input_paths)