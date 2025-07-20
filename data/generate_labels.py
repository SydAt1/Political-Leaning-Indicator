import pandas as pd
import os

# Load the history data
cleaned_path = "data/processed/cleaned_history.csv"
history_path = 'data/raw/user1_history.csv'
history_path = cleaned_path if os.path.exists(cleaned_path) else raw_path

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
labels_df.to_csv("data/labels.csv", index=False)