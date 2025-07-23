from src.scrape import chrome_history, firefox_history
from data import generate_labels
from src.nlp.preprocess import preprocess_history
from src.nlp.vectorizer import vectorize_text
from src.models.predictor import predict
from src.models.train_classifier import train_classifier  # Import the training function
import os
import pickle

def main():
    # Prompt user for browser choice
    browser = input("Choose browser to scrape history (chrome/firefox): ").strip().lower()
    if browser == "chrome":
        print("Scraping Chrome browser history...")
        raw_paths = chrome_history.main()
    elif browser == "firefox":
        print("Scraping Firefox browser history...")
        raw_paths = firefox_history.main()
    else:
        print("Invalid choice. Please enter 'chrome' or 'firefox'.")
        return

    # Validate raw_paths
    print(f"Raw paths: {raw_paths}")  # Debug print
    if not raw_paths or not os.path.exists(raw_paths):
        raise ValueError(f"Invalid or missing raw history path: {raw_paths}")

    # Preprocess the raw history data
    print("Preprocessing raw data...")
    cleaned_df = preprocess_history(raw_paths)  # Use raw_paths as input

    # Generate labels for preprocessed data
    print("Generating labels...")
    labeled_paths = generate_labels.main(cleaned_df)  # Pass DataFrame

    # Vectorize the cleaned and labeled data and save TF-IDF matrix
    print("Vectorizing data...")
    tfidf_matrix, vectorizer = vectorize_text(cleaned_df)
    tfidf_path = 'data/processed/tfidf_features.pkl'
    os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"Saved TF-IDF matrix to {tfidf_path}")

    # Train the Logistic Regression model
    print("Training logistic regression model...")
    model, accuracy = train_classifier(tfidf_path=tfidf_path, labels_path='data/labels.csv')
    print(f"Model trained with accuracy: {accuracy:.2f}")

    # Save the model (train_classifier already does this, but we can confirm)
    model_path = 'models/logistic_regression.pkl'
    print(f"Model saved to {model_path}")

    # Make prediction using the trained model
    print("Making prediction...")
    prediction = predict(tfidf_matrix)
    print(f"Political leaning prediction: {prediction}")

    print("Pipeline completed.")

if __name__ == "__main__":
    main()