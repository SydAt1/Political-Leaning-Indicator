import pandas as pd
import os

def main(raw_paths=None):
    """
    Generate labels for browser history data and save to a CSV file.
    
    Args:
        raw_paths (str or list, optional): Path(s) to raw history data. Defaults to None,
                                          using predefined paths if not provided.
    
    Returns:
        str: Path to the generated labels CSV file.
    """
    # Load the history data
    cleaned_path = "data/processed/cleaned_history.csv"
    history_path = cleaned_path if os.path.exists(cleaned_path) else "data/raw/user1_history.csv"

    df = pd.read_csv(history_path)

    # Keyword-based labeling function
    def assign_label(title, url):
        title = title.lower()
        if any(keyword in title for keyword in ["trump", "republican", "conservative", "gop"]):
            return 1  # Republican-leaning
        elif any(keyword in title for keyword in ["democrat", "liberal", "biden", "newsom", "climate"]):
            return 0  # Democrat-leaning
        else:
            return 0  # Default to neutral/Democrat-leaning

    # Apply labeling
    df['label'] = df['cleaned_title'].fillna(df['title']).apply(lambda x: assign_label(x, ""))

    # Save to labels.csv
    labels_df = df[['label']]
    output_path = "data/labels.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels_df.to_csv(output_path, index=False)

    return output_path  # Return the path to the labeled data

if __name__ == "__main__":
    # For standalone testing
    main()