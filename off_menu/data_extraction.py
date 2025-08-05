# =========================================================================
# 1. Imports
# =========================================================================
import os
import time
import random

import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import random

from .config import (episodes_list_url,
                    restaurants_url
)

from .utils import (save_text_to_file,
                    extract_html,
                    try_read_parquet)

# =========================================================================
# 2. Configuration (paths, constants, etc.)
# =========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
test_temp_dir = os.path.join(project_root, 'data/test_temp')
raw_data_path = os.path.join(project_root, 'data/raw')
processed_data_path = os.path.join(project_root, 'data/processed')

# =========================================================================
# 3. Helper Functions
# =========================================================================

def _save_transcripts_html(eps_dataframe, directory):
    """
    Iterates through a DataFrame of episodes, downloads the HTML content from
    the episode URL, and saves it to a specified directory.
    
    Skips files that already exist and includes a random delay to be
    polite to the server.
    
    Args:
        eps_dataframe (pd.DataFrame): DataFrame containing episode metadata
                                      (including 'episode_number' and 'url').
        directory (str): The directory to save the HTML files to.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for index, row in eps_dataframe.iterrows():
        episode_num = row['episode_number'] 
        episode_url = row['url']
        filename = f'ep_{episode_num}.html'
        filepath = os.path.join(directory, filename)

        # Skip episodes that already exist
        if os.path.exists(filepath):
            print(f"  Skipping Episode {episode_num}: File already exists at {filepath}")
            continue

        # Delay to be polite to the server and avoid 429 errors
        sleep_time = random.uniform(1, 3) # Sleep for 1 to 3 seconds
        time.sleep(sleep_time)

        html_content_str = extract_html(episode_url)

        # Check for None before attempting to save
        # The extract_html function returns None on failure (like a 429 error)
        if html_content_str:
            save_text_to_file(html_content_str, filename, directory)
        else:
            print(f"  Skipping save for Episode {episode_num} due to failed extraction.")

# =========================================================================
# 4. Main Logic Functions
# =========================================================================

def extract_and_save_html(
    site_url,
    output_html_filepath 
):
    """
    Downloads HTML content from a given URL and saves it to a file.
    
    Args:
        site_url (str): The URL to scrape.
        output_html_filepath (str): The full path to the output HTML file.
    """
    html_content = extract_html(site_url)
    
    if html_content:
        directory, filename = os.path.split(output_html_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_text_to_file(html_content, filename, directory)


def extract_and_save_transcripts_html(input_dataframe_filepath: str,
    output_html_directory: str
    ) -> None:
    """
    Loads episode metadata and uses it to download and save transcript HTML files.

    Args:
        input_dataframe_filepath (str): The path to the parquet file containing
                                        episode metadata.
        output_html_directory (str): The directory where the transcript
                                     HTML files will be saved.
    """
    ep_meta_and_mentions_df = try_read_parquet(input_dataframe_filepath)
    _save_transcripts_html(ep_meta_and_mentions_df, output_html_directory)


# =========================================================================
# 5. Script exectuion
# This section contains script that runs only when this script is run directly when it is open (not when called by another script)
# This will contain a smaller model of the processes, so we can test before implementing in main
# =========================================================================

if __name__ == "__main__":

# -------------------------------------------------------------------------
# Episode extraction
# -------------------------------------------------------------------------
    save_text_to_file(extract_html(episodes_list_url), 'episodes.html', test_temp_dir)

# -------------------------------------------------------------------------
# Restaurants extraction
# -------------------------------------------------------------------------

# Extract restaurants site html and store
    save_text_to_file(extract_html(restaurants_url), 'restaurants_site.html', test_temp_dir)

# -------------------------------------------------------------------------
# Transcripts extraction
# -------------------------------------------------------------------------
# Note - we use dummy data here both to avoid having to depend on data processing, and bc purpose of testing

    # Opening dummy data (head of full dataframe) and testing extraction
    ep_meta_and_mentions_head_path = os.path.join(test_temp_dir, 'ep_meta_and_mentions_head.parquet')
    try:
        ep_meta_and_mentions_head_df = pd.read_parquet(ep_meta_and_mentions_head_path)
        #Testing on dummy data
        _save_transcripts_html(ep_meta_and_mentions_head_df, test_temp_dir)
    except FileNotFoundError:
        print(f"Error: The file was not found at {ep_meta_and_mentions_head_path}. Did it save correctly?")
