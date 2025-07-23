import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from shutil import copyfile
import sys

# Adjust sys.path to include src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import logger setup
from utils.logger import setup_logger

# Set up logging
logger = setup_logger('chrome_history')

def get_chromium_history_path():
    """Determine Chrome or Brave history file path based on OS and browser."""
    base_paths = {
        'chrome': {
            'nt': os.path.join(os.getenv('LOCALAPPDATA'), r'Google\Chrome\User Data\Default\History'),
            'posix': os.path.join(os.path.expanduser('~'), '.config/google-chrome/Default/History')
        },
        'brave': {
            'nt': os.path.join(os.getenv('LOCALAPPDATA'), r'BraveSoftware\Brave-Browser\User Data\Default\History'),
            'posix': os.path.join(os.path.expanduser('~'), '.config/BraveSoftware/Brave-Browser/Default/History')
        }
    }
    for browser, paths in base_paths.items():
        path = paths.get(os.name)
        if path and os.path.exists(path):
            logger.info(f"Found {browser} history file at {path}")
            return path, browser
    raise FileNotFoundError("No Chrome or Brave history file found")

def copy_history_file(history_path, temp_path='temp_history.db'):
    """Copy history file to avoid locking issues."""
    try:
        copyfile(history_path, temp_path)
        logger.info(f"Copied history file to {temp_path}")
        return temp_path
    except FileNotFoundError:
        logger.error(f"History file not found at {history_path}")
        raise
    except PermissionError:
        logger.error("Permission denied accessing history file")
        raise

def extract_chromium_history(temp_path, output_path='data/raw/userb_history.csv', 
                           start_date=None, end_date=None, keywords=None, limit=300):
    """Extract the 300 most recent history entries from Chrome/Brave SQLite database."""
    try:
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()
        query = """
        SELECT urls.url, urls.title, urls.last_visit_time
        FROM urls
        JOIN visits ON urls.id = visits.url
        WHERE 1=1
        """
        params = []
        if start_date:
            start_timestamp = int((start_date - datetime(1601, 1, 1)).total_seconds() * 1_000_000)
            query += " AND urls.last_visit_time >= ?"
            params.append(start_timestamp)
        if end_date:
            end_timestamp = int((end_date - datetime(1601, 1, 1)).total_seconds() * 1_000_000)
            query += " AND urls.last_visit_time <= ?"
            params.append(end_timestamp)
        if keywords:
            keyword_conditions = " OR ".join(["urls.url LIKE ? OR urls.title LIKE ?" for _ in keywords])
            query += f" AND ({keyword_conditions})"
            for keyword in keywords:
                params.extend([f"%{keyword}%", f"%{keyword}%"])
        query += " ORDER BY urls.last_visit_time DESC LIMIT ?"
        params.append(limit)  # Add limit parameter
        cursor.execute(query, params)
        history_data = cursor.fetchall()
        columns = ['url', 'title', 'last_visit_time']
        df = pd.DataFrame(history_data, columns=columns)

        # Convert Chrome/Brave timestamp (microseconds since 1601-01-01) to datetime
        EPOCH = datetime(1601, 1, 1)
        df['last_visit_time'] = df['last_visit_time'].apply(
            lambda x: EPOCH + timedelta(microseconds=x) if pd.notna(x) else pd.NaT
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} recent history records to {output_path}")
        return output_path
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        raise
    finally:
        conn.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Removed temporary file {temp_path}")

def main(start_date=None, end_date=None, keywords=None):
    """Main function to orchestrate history scraping with optional filters."""
    try:
        logger.info("Please ensure Chrome or Brave is closed before running.")
        history_path, browser = get_chromium_history_path()
        temp_path = f'temp_{browser}_history.db'
        temp_db = copy_history_file(history_path, temp_path)
        output_path = f'data/raw/user{browser[0]}_history.csv'
        extract_chromium_history(temp_path, output_path, start_date, end_date, keywords)
        logger.info(f"{browser.capitalize()} history extraction completed")
        return output_path
    except Exception as e:
        logger.error(f"Failed to extract history: {e}")
        raise

if __name__ == "__main__":
    print("This script requires explicit user consent to access browser history.")
    consent = input("Do you consent to extracting your Chrome/Brave history? (yes/no): ").lower()
    if consent == 'yes':
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 7, 19)
        keywords = ['politics', 'news']
        main(start_date, end_date, keywords)
    else:
        logger.warning("User did not provide consent. Exiting.")
        print("Consent not provided. Exiting.")