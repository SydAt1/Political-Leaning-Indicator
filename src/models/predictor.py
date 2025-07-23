import os
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'logistic_regression.pkl')

def predict(tfidf_matrix):
    """
    Predict political leaning based on TF-IDF matrix using a pre-trained model.
    
    Args:
        tfidf_matrix (scipy.sparse.csr_matrix or numpy.ndarray): TF-IDF representation of text data.
    
    Returns:
        str: Predicted political leaning ("Republican" or "Democrat") with confidence.
    """
    # Load the pre-trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    # Ensure tfidf_matrix is in the correct format (dense array if sparse)
    if hasattr(tfidf_matrix, 'toarray'):
        X = tfidf_matrix.toarray()
    else:
        X = np.asarray(tfidf_matrix)
    
    # Make prediction
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    confidence = np.max(probability, axis=1)  # Highest probability for the predicted class
    
    # Map prediction to label
    label = "Republican" if prediction[0] == 1 else "Democrat"
    
    # Return result with confidence (average confidence for multiple samples, if applicable)
    return f"{label} (Confidence: {confidence[0]:.2f})"

# Optional: Add a function to train or update the model if needed
def train_model(X, y, save_path=MODEL_PATH):
    """
    Train a new Logistic Regression model and save it.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target labels.
        save_path (str): Path to save the trained model.
    """
    model = LogisticRegression()
    model.fit(X, y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")