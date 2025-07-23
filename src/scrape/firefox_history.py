import sqlite3
import os
import pandas as pd
from datetime import datetime
import logging
from shutil import copyfile
import sys

# Adjust sys.path to include src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import logger setup
from utils.logger import setup_logger

# Set up logging
logger = setup_logger('firefox_history')

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

def extract_firefox_history(temp_path, output_path='data/raw/user2_history.csv',
                           start_date=None, end_date=None, keywords=None, limit=300):
    """Extract the 300 most recent history entries from Firefox SQLite database."""
    try:
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()

        query = """
        SELECT moz_places.url, moz_places.title, moz_places.last_visit_date
        FROM moz_places
        WHERE moz_places.last_visit_date IS NOT NULL
        """
        params = []

        # Apply time filter (timestamps are in microseconds since 1970)
        if start_date:
            start_ts = int(start_date.timestamp() * 1_000_000)
            query += " AND moz_places.last_visit_date >= ?"
            params.append(start_ts)
        if end_date:
            end_ts = int(end_date.timestamp() * 1_000_000)
            query += " AND moz_places.last_visit_date <= ?"
            params.append(end_ts)

        # Keyword filtering
        if keywords:
            query += " AND (" + " OR ".join(["moz_places.url LIKE ? OR moz_places.title LIKE ?" for _ in keywords]) + ")"
            for kw in keywords:
                params.extend([f"%{kw}%", f"%{kw}%"])

        query += " ORDER BY moz_places.last_visit_date DESC LIMIT ?"
        params.append(limit)  # Add limit parameter

        cursor.execute(query, params)
        history_data = cursor.fetchall()

        columns = ['url', 'title', 'last_visit_time']
        df = pd.DataFrame(history_data, columns=columns)

        # Convert Firefox timestamp (microseconds since 1970-01-01) to datetime
        df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], unit='us')

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
    """Main function to orchestrate Firefox history scraping."""
    try:
        logger.info("Please ensure Firefox is closed before running.")
        history_path = get_firefox_history_path()
        temp_path = 'temp_firefox_history.db'
        temp_db = copy_history_file(history_path, temp_path)
        output_path = 'data/raw/user2_history.csv'
        extract_firefox_history(temp_db, output_path, start_date, end_date, keywords)
        logger.info("Firefox history extraction completed")
        return output_path
    except Exception as e:
        logger.error(f"Failed to extract Firefox history: {e}")
        raise

if __name__ == "__main__":
    print("This script requires explicit user consent to access browser history.")
    consent = input("Do you consent to extracting your Firefox history? (yes/no): ").lower()
    if consent == 'yes':
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 7, 19)
        keywords = ['politics', 'news']
        main(start_date, end_date, keywords)
    else:
        logger.warning("User did not provide consent. Exiting.")
        print("Consent not provided. Exiting.")