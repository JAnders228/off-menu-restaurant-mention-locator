# main.py

import os
import sys
import pandas as pd

# --- 1. Import Pipeline Functions ---
from off_menu.data_extraction import (
    extract_and_save_html,
    extract_and_save_transcripts_html
)
from off_menu.data_processing import (
    create_numbers_names_dict_from_html,
    create_numbers_names_df_from_dict,
    create_urls_and_save_to_numbers_names_df,
    create_mentions_by_res_name_dict,
    create_return_exploded_res_mentions_df,
    combine_save_mentions_ep_metadata_dfs,
    extract_clean_text_and_periodic_timestamps,
    combine_timestamps_and_metadata,
    find_top_match_and_timestamps
)
from off_menu.utils import(
    try_read_parquet
)

# --- 2. Define Your Paths and Parameters (Hardcoded for speed, refine later) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ANALYTICS_DATA_DIR = os.path.join(DATA_DIR, 'analytics')

# --- Specific File Paths ---
INITIAL_EPISODE_LIST_HTML_PATH = os.path.join(RAW_DATA_DIR, 'off_menu_episodes_page.html') 

# --- Parameters ---
# FUZZY_MATCH_THRESHOLD = 90
from off_menu.config import (
    episodes_list_url,
    transcript_base_url,
    restaurants_url)

# --- 3. Main Orchestration Function ---
def main():
    print("--- Starting Off Menu Podcast Data Pipeline ---")

    try:
        # Create necessary directories
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(ANALYTICS_DATA_DIR, exist_ok=True)
        print("Setup: All necessary directories ensured.")

        # =====================================================================
        # STAGE 1: Initial Data Extraction (episodes metadata, restaurant mentions)
        # Goal: Get episodes metadata and restaurant mentions HTML
        # =====================================================================
        print("\n--- STAGE 1: Downloading HTML from episodes and restaurants sites ---")
        episodes_html_filepath = os.path.join(RAW_DATA_DIR, 'episodes.html')
        restaurants_html_filepath = os.path.join(RAW_DATA_DIR, 'restaurants.html')

        extract_and_save_html(episodes_list_url, episodes_html_filepath)
        extract_and_save_html(restaurants_url, restaurants_html_filepath)

        print("STAGE 1 Complete: Initial HTML downloaded.")

        # =====================================================================
        # STAGE 2: Preliminary Processing (Generate Full URLs for Transcripts)
        # Goal: Take basic metadata, clean guest names, and generate the *exact*
        #       URLs needed for downloading each transcript's HTML.
        # =====================================================================
        print("\n--- STAGE 2: Processing Episode Metadata & Generating Transcript URLs ---")
        processed_metadata_filepath_for_saving = os.path.join(PROCESSED_DATA_DIR, 'num_name_url_df.parquet')

        # Step 2.1 Create numbers and names dictionary from html

        numbers_names_dict = create_numbers_names_dict_from_html(episodes_html_filepath)

        # Step 2.2 Create numbers and names dataframe from numbers and names dictionary

        numbers_names_df = create_numbers_names_df_from_dict(numbers_names_dict)

        # Step 2.3 Create URL's and add to the datframe, save dataframe

        num_name_url_df = create_urls_and_save_to_numbers_names_df(numbers_names_df, processed_metadata_filepath_for_saving)

        print("STAGE 2 Complete: numbers, names, urls dataframe saved.")

        # =====================================================================
        # STAGE 3: Dependent Extraction (Download Full HTML Transcripts)
        # Goal: Use the processed metadata (with full URLs) to download the transcripts.
        # =====================================================================
        print("\n--- STAGE 3: Downloading Full HTML Transcripts ---")
        
        raw_transcripts_output_dir = os.path.join(RAW_DATA_DIR, 'transcripts_htmls')
        extract_and_save_transcripts_html(processed_metadata_filepath_for_saving, raw_transcripts_output_dir)

        print("STAGE 3 Complete: Full HTML transcripts downloaded.")

        # =====================================================================
        # STAGE 4: Processing restaurant mentions (which episodes mention which, if any, restaurants)
        # Goal: Process the downloaded raw HTML from the restaurants site, and combine with numbers, names, url df
        # =====================================================================
        print("\n--- STAGE 4: Processing restaurant mentions from restaurants HTML ---")
        full_episodes_data_path = os.path.join(ANALYTICS_DATA_DIR, 'full_episodes_data_df.parquet')

        # Step 4.1 Create dict of mentions with res name as keys and list of guests who mention as values
        guests_who_mention_res_by_res_name_dict = create_mentions_by_res_name_dict(restaurants_html_filepath)
        # Step 4.2 Convert into exploded dataframe (one line per guest who mentions)
        exploded_res_mentions_df = create_return_exploded_res_mentions_df(guests_who_mention_res_by_res_name_dict)
        # Step 4.3 Combine with numbers, names, url dataframe and save
        combine_save_mentions_ep_metadata_dfs(exploded_res_mentions_df, processed_metadata_filepath_for_saving, full_episodes_data_path)

        print("STAGE 4 Complete: Restaurant mentions merged with metadata to produce full_episode_data_df")
        # =====================================================================
        # STAGE 5: Collate clean transcript strings and timestamps for each episode
        # Goal: Create a dataframe of all the cleaned transcript strings, and timestamps (time, index) for all of the transcripts, by episode number
        # =====================================================================
        print("\n--- STAGE 5: Collating the timestamps for each episode ---")
        full_episodes_metadata_path = full_episodes_data_path
        transcripts_dir = raw_transcripts_output_dir
        output_filepath = os.path.join(PROCESSED_DATA_DIR, 'cleaned_transcript_timestamp_df.parquet')

        extract_clean_text_and_periodic_timestamps(full_episodes_metadata_path, transcripts_dir, output_filepath)
            
        print("STAGE 5 Complete: Clean transcripts and timestamps collated.")

        # =====================================================================
        # STAGE 6: First run of fuzzymatching => full restaurant mentions/easy wins (MVP)
        # Goal: Match full names (e.g. 'Paul Bakery') and store quote, index and nearest timestamp in a dataframe, for MVP of easy matches
        # =====================================================================
        print("\n--- STAGE 6: Fuzzymatch full restaurant mentions (easy wins) ---")

        cleaned_transcript_timestamps_filepath = os.path.join(PROCESSED_DATA_DIR, 'cleaned_transcript_timestamp_df.parquet')
        full_episodes_metadata_path = full_episodes_data_path

        combined_df = combine_timestamps_and_metadata(cleaned_transcript_timestamps_filepath, full_episodes_metadata_path)
        easy_win_mention_search_df = find_top_match_and_timestamps(combined_df)

        easy_wins_mention_search_path = os.path.join(PROCESSED_DATA_DIR, 'easy_win_mention_search_df.parquet')
        easy_win_mention_search_df.to_parquet(easy_wins_mention_search_path, index=False)
        
        print("STAGE 6 Complete: First run of fuzzymatches complete.")

        # =====================================================================
        # STAGE 7: (Optional) Final Output / Analysis
        # =====================================================================
        print("\n--- STAGE 7: Final Output / Analysis ---")
        # You might load aggregated_output_path here to do some quick analysis
        easy_win_mention_search_df = try_read_parquet(easy_wins_mention_search_path)

        print("\n--- Head of the easy wins mention search dataframe ---")
        print(easy_win_mention_search_df.head(10))
        
        print("STAGE 7 Complete: Final output reviewed.")
        print("\n--- Pipeline Finished Successfully! ---")

    except Exception as e:
        print(f"\n!!! PIPELINE FAILED !!!")
        print(f"Error: {e}")
        print("Please check the error message above and your logs/intermediate files.")
        sys.exit(1) # Exit with an error code to indicate failure

# --- Main execution block ---
if __name__ == "__main__":
    main()