import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from utils.logger import setup_logger
import pickle
import sys
from scipy.sparse import csr_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up logging
logger = setup_logger('model_train')

def train_classifier(tfidf_path='data/processed/tfidf_features.pkl', labels_path='data/labels.csv', 
                    model_path='models/logistic_regression.pkl'):
    """Train a Logistic Regression classifier and save the model."""
    try:
        # Load TF-IDF features as a sparse matrix
        with open(tfidf_path, 'rb') as f:
            X = pickle.load(f)
        if not isinstance(X, csr_matrix):
            raise ValueError(f"Expected a sparse matrix, got {type(X)}")
        logger.info(f"Loaded TF-IDF features from {tfidf_path} with shape {X.shape}")

        # Load labels
        labels_df = pd.read_csv(labels_path)
        if 'label' not in labels_df.columns:
            raise ValueError("Labels file must contain a 'label' column")
        y = labels_df['label'].values
        if len(y) != X.shape[0]:
            raise ValueError("Mismatch between number of samples in TF-IDF and labels")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Split data into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples)")

        # Train Logistic Regression (supports sparse input)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy on test set: {accuracy:.2f}")

        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {model_path}")

        return model, accuracy

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_classifier()