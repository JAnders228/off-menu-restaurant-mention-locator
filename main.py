# main.py

import os
import sys
import pandas as pd
import traceback

# --- 1. Import Pipeline Functions ---
from off_menu.data_extraction import (
    extract_and_save_html,
    orchestrate_scraper_legacy # replaces extract_and_save_transcripts_html,
)
from off_menu.data_processing import (
    create_tuple_inc_ep_slugs_guests_list_from_html, # replace create_numbers_names_dict_from_html
    create_slugs_guests_df_from_list_of_dict, # replaces create_numbers_names_df_from_dict
    create_urls_and_save_to_slugs_guests_df, # replaces create_urls_and_save_to_numbers_names_df
    create_mentions_by_res_name_dict,
    create_return_exploded_res_mentions_df,
    combine_save_mentions_and_ep_metadata_dfs,
    extract_save_clean_text_and_periodic_timestamps, #replaces extract_clean_text_and_periodic_timestamps
    combine_timestamps_and_metadata,
    find_top_match_and_timestamps,
    get_unprocessed_episodes,
    parse_restaurants_using_user_cleaners_v3,
    exact_merge_restaurants
    
)
from off_menu.utils import try_read_parquet, try_read_html_string_from_filepath

# --- 2. Define Your Paths and Parameters (Hardcoded for speed, refine later) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
ANALYTICS_DATA_DIR = os.path.join(DATA_DIR, "analytics")

# --- Specific File Paths ---
INITIAL_EPISODE_LIST_HTML_PATH = os.path.join(
    RAW_DATA_DIR, "off_menu_episodes_page.html"
)

# --- Parameters ---
# FUZZY_MATCH_THRESHOLD = 90
from off_menu.config import episodes_list_url, transcript_base_url, restaurants_url


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
        episodes_html_filepath = os.path.join(RAW_DATA_DIR, "episodes.html")
        restaurants_html_filepath = os.path.join(RAW_DATA_DIR, "restaurants.html")

        extract_and_save_html(episodes_list_url, episodes_html_filepath)
        extract_and_save_html(restaurants_url, restaurants_html_filepath)

        print("STAGE 1 Complete: Initial HTML downloaded.")

        # =====================================================================
        # STAGE 2: Preliminary Processing (Generate Full URLs for Transcripts)
        # Goal: Take basic metadata, clean guest names, and generate the exact
        # URLs needed for downloading each transcript's HTML.
        # Also, generate a dataframe with restaurant mentions data (regions) for use in ordering output
        # =====================================================================
        print(
            "\n--- STAGE 2: Processing Episode Metadata & Generating Transcript URLs ---"
        )
        processed_metadata_filepath_for_saving = os.path.join(
            PROCESSED_DATA_DIR, "slugs_names_urls_df.parquet"
        )

        # Step 2.1 Create tuple containing two lists
        # The first list is a list of dicts of raw titles, slugs and guest names
        # Which will be used for onwards processing into a full metadata dataframe
        # The second is exceptions (e.g. best of) which will not be included 

        episodes_html_str = try_read_html_string_from_filepath(episodes_html_filepath)

        title_slugs_guestnames_list = create_tuple_inc_ep_slugs_guests_list_from_html(episodes_html_str)[0]

        # Step 2.2 Create titles, slugs, guest names dataframe from list of dicts of raw titles, slugs and guest names
        title_slugs_guestnames_df = create_slugs_guests_df_from_list_of_dict(title_slugs_guestnames_list)

        # Step 2.3 Create URL's and add to the datframe, save dataframe

        create_urls_and_save_to_slugs_guests_df(
            title_slugs_guestnames_df, processed_metadata_filepath_for_saving
        )
        titles_slugs_guestnames_urls_df = try_read_parquet(processed_metadata_filepath_for_saving)

        # Step 2.4 Create a dataframe of restaurants including guests who mention, and their regions

        restaurants_and_regions_filepath_for_saving = os.path.join(
            PROCESSED_DATA_DIR, "res_and_regions_df.parquet"
        )
        restaurants_html = try_read_html_string_from_filepath(restaurants_html_filepath)

        restaurants_and_regions_df = parse_restaurants_using_user_cleaners_v3(restaurants_html)

        restaurants_and_regions_df.to_parquet(restaurants_and_regions_filepath_for_saving)

        print("STAGE 2 Complete: raw titles, slugs, guestnames, urls dataframe saved. Restaurants and regions dataframe saved")

        # =====================================================================
        # STAGE 3: Dependent Extraction (Download Full HTML Transcripts)
        # Goal: Use the processed metadata (with full URLs) to download the transcripts.
        # =====================================================================
        print("\n--- STAGE 3: Downloading Full HTML Transcripts ---")

        # Consider adding function to check for old style transcripts and update status accordingly

        raw_transcripts_output_dir = os.path.join(RAW_DATA_DIR, "transcripts_htmls")
        
        orchestrate_scraper_legacy(
            df = titles_slugs_guestnames_urls_df, 
            base_url=transcript_base_url, 
            out_dir = raw_transcripts_output_dir,
            max_attempts_per_url= 5, 
            backoff_base= 1,
            max_workers= 3,
            legacy_dir=raw_transcripts_output_dir
        )

        print("STAGE 3 Complete: HTML transcripts attemped, see logger above for success rate")

        # =====================================================================
        # STAGE 4: Processing restaurant mentions (which episodes mention which, if any, restaurants)
        # Goal: Process the downloaded raw HTML from the restaurants site, and combine with numbers, names, url df
        # =====================================================================
        print("\n--- STAGE 4: Processing restaurant mentions from restaurants HTML ---")
        episodes_metadata_and_mentions_path = os.path.join(
            ANALYTICS_DATA_DIR, "episodes_metadata_and_mentions_df.parquet"
        )

        # Step 4.1 Create dict of mentions with res name as keys and list of guests who mention as values
        guests_who_mention_res_by_res_name_dict = create_mentions_by_res_name_dict(
            restaurants_html_filepath
        )
        # Step 4.2 Convert into exploded dataframe (one line per guest who mentions)
        exploded_res_mentions_df = create_return_exploded_res_mentions_df(
            guests_who_mention_res_by_res_name_dict
        )
        # Step 4.3 Combine with numbers, names, url dataframe and save
        combine_save_mentions_and_ep_metadata_dfs(
            exploded_res_mentions_df,
            processed_metadata_filepath_for_saving,
            episodes_metadata_and_mentions_path,
        )
        print("Episodes metadata and mentions df:")
        print(try_read_parquet(episodes_metadata_and_mentions_path).head())
        print(
            "STAGE 4 Complete: Restaurant mentions merged with metadata to produce episodes_metadata_and_mentions_df"
        )
        # =====================================================================
        # STAGE 5: Collate clean transcript strings and timestamps for each episode
        # Goal: Create a dataframe of all the cleaned transcript strings, and timestamps (time, index) for all of the transcripts, by episode number
        # =====================================================================
        print("\n--- STAGE 5: Collating the timestamps for each episode ---")

        transcripts_dir = raw_transcripts_output_dir

        timestamps_transcripts_output_filepath = os.path.join(
            PROCESSED_DATA_DIR, "cleaned_transcripts_timestamps_df.parquet"
        )

        # Check for processed episodes
        episodes_for_timestamping = get_unprocessed_episodes(
        episodes_metadata_and_mentions_path,
        timestamps_transcripts_output_filepath 
        )
        # Save unprocessed episodes (to be processed) into dataframe
        episodes_for_timestamping_path = os.path.join(
            PROCESSED_DATA_DIR, "episodes_for_timestamping_df.parquet"
        )
        episodes_for_timestamping.to_parquet(episodes_for_timestamping_path)

        # Extract new timestamps
        extract_save_clean_text_and_periodic_timestamps(
            episodes_for_timestamping_path, transcripts_dir, timestamps_transcripts_output_filepath
        )

        print("STAGE 5 Complete: Clean transcripts and timestamps collated.")

        # =====================================================================
        # STAGE 6: First run of fuzzymatching => full restaurant mentions/easy wins (MVP) => Merged with restaurant regions data for output
        # Goal: Match full names (e.g. 'Paul Bakery') and store quote, index and nearest timestamp in a dataframe, for MVP of easy matches
        # Goal: Merge this dataframe with restaurants and regions data ready for output
        # =====================================================================
        print("\n--- STAGE 6: Fuzzymatch full restaurant mentions (easy wins) and merge with restaurant regions data ---")

        # Stage 6.1 Fuzzy matching
        cleaned_transcript_timestamps_filepath = os.path.join(
            PROCESSED_DATA_DIR, "cleaned_transcripts_timestamps_df.parquet"
        )

        # Note: cleaned_transcrtip_timestamps_df has metadata and transcripts/timestamps
        # So, no need to combine with metadat
        # Combination function results in two restaurant_mentions columns, restaurants_mentions_x and y
        # Which then cannot be read by the matching function
        cleaned_transcript_timestamps_df = try_read_parquet(cleaned_transcript_timestamps_filepath)

        print("attempting easy win mention search")
        easy_win_mention_search_df = find_top_match_and_timestamps(cleaned_transcript_timestamps_df)

        easy_wins_mention_search_path = os.path.join(
            PROCESSED_DATA_DIR, "easy_win_mention_search_df.parquet"
        )
        print(f"saving easy win mentions to filepath {easy_wins_mention_search_path}")
        easy_win_mention_search_df.to_parquet(
            easy_wins_mention_search_path, index=False
        )

        # Stage 6.2 Merging easy wins with retsaurants and regions data

        matched, unmatched, report = exact_merge_restaurants(restaurants_and_regions_df, easy_win_mention_search_df)

        print(f"report: {report}")
        print(f"\n unmatched df head: {unmatched.head()}")
        print(f"\n matched df head: {matched.head()}")
        print(f"\n matched df cols: {matched.columns}")

        print(unmatched[['Restaurant','Episode ID','Mention text']].head(20).to_string())

        top_mentions_with_regions_path = os.path.join(PROCESSED_DATA_DIR, "top_mentions_with_regions.csv")

        matched.to_csv(top_mentions_with_regions_path)

        print("STAGE 6 Complete: First run of fuzzymatches complete. Dataframe merged with restaurant region data")

        # =====================================================================
        # STAGE 7: (Optional) Final Output / Analysis
        # =====================================================================
        print("\n--- STAGE 7: Final Output / Analysis ---")
        # You might load aggregated_output_path here to do some quick analysis
        easy_win_mention_search_df = try_read_parquet(easy_wins_mention_search_path)

        print("\n--- Head of the matched easy wins mention search dataframe ---")
        print(matched.head(10))

        print("STAGE 7 Complete: Final output reviewed.")
        print("\n--- Pipeline Finished Successfully! ---")

    except Exception as e:
        print(f"\n!!! PIPELINE FAILED !!!")
        print(f"Error: {e}")
        traceback.print_exc()
        print("Please check the error message above and your logs/intermediate files.")
        sys.exit(1)  # Exit with an error code to indicate failure


# --- Main execution block ---
if __name__ == "__main__":
    main()
