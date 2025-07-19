import sqlite3
import os
import pandas as pd
import logging
from shutil import copyfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/scrape_history.log'
)
logger = logging.getLogger(__name__)

def get_firefox_history_path():
    """Determine Firefox history file path by reading profiles.ini."""
    profile_dir = os.path.join(os.path.expanduser('~'), '.mozilla/firefox')
    profiles_ini = os.path.join(profile_dir, 'profiles.ini')
    try:
        with open(profiles_ini, 'r') as f:
            for line in f:
                if 'Path=' in line:
                    profile = line.strip().split('=')[1]
                    history_path = os.path.join(profile_dir, profile, 'places.sqlite')
                    if os.path.exists(history_path):
                        logger.info(f"Found Firefox history file at {history_path}")
                        return history_path
        raise FileNotFoundError("Firefox profile not found")
    except FileNotFoundError:
        logger.error(f"profiles.ini not found at {profiles_ini}")
        raise

def copy_history_file(history_path, temp_path='temp_firefox_history.db'):
    """Copy Firefox history file to avoid locking issues."""
    try:
        copyfile(history_path, temp_path)
        logger.info(f"Copied history file to {temp_path}")
        return temp_path
    except FileNotFoundError:
        logger.error(f"History file not found at {history_path}")
        raise
    except PermissionError:
        logger.error("Permission denied accessing Firefox history file")
        raise

def extract_firefox_history(temp_path, output_path='data/raw/user2_history.csv'):
    """Extract history from Firefox SQLite database and save to CSV."""
    try:
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()

        query = """
        SELECT moz_places.url, moz_places.title, moz_places.last_visit_date
        FROM moz_places
        WHERE moz_places.last_visit_date IS NOT NULL
        ORDER BY moz_places.last_visit_date DESC
        """
        cursor.execute(query)
        history_data = cursor.fetchall()

        columns = ['url', 'title', 'last_visit_time']
        df = pd.DataFrame(history_data, columns=columns)

        # Convert Firefox timestamp (microseconds since 1970-01-01) to datetime
        df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], unit='us')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved history to {output_path} with {len(df)} records")

        return df

    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        raise
    finally:
        conn.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed temporary file {temp_path}")

def main():
    """Main function to orchestrate history scraping."""
    try:
        logger.info("Please ensure Firefox is closed before running.")
        history_path = get_firefox_history_path()
        temp_path = 'temp_firefox_history.db'
        temp_db = copy_history_file(history_path, temp_path)
        extract_firefox_history(temp_db, 'data/raw/user2_history.csv')
        logger.info("Firefox history extraction completed")

    except Exception as e:
        logger.error(f"Failed to extract history: {e}")
        raise

if __name__ == "__main__":
    print("This script requires explicit user consent to access browser history.")
    consent = input("Do you consent to extracting your Firefox history? (yes/no): ").lower()
    if consent == 'yes':
        main()
    else:
        logger.warning("User did not provide consent. Exiting.")
        print("Consent not provided. Exiting.")