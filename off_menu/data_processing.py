# =========================================================================
# 1. Imports
# =========================================================================
import os

import pandas as pd
from bs4 import BeautifulSoup
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import unicodedata

from .utils import try_read_html_string_from_filepath, try_read_parquet
from .config import transcript_base_url
from .data_extraction import orchestrate_scraper

# =========================================================================
# 2. Configuration (paths, constants, etc.)
# =========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
test_temp_dir = os.path.join(project_root, "data", "test_temp")
raw_data_path = os.path.join(project_root, "data", "raw")
processed_data_path = os.path.join(project_root, "data/processed")
episodes_html_filepath = os.path.join(
    project_root, "data", "test_temp", "episodes.html"
)

# =========================================================================
# 3. Helper Functions
# =========================================================================

# -------------------------------------------------------------------------
# Episodes processing
# -------------------------------------------------------------------------

# Helper function to create slugs, which will be used as unique ep identifiers now that numbers 
# are no longer give, and will form the end of URL's
# Used in create_ep_slugs_and_guests_from_html
def slugify(text: str) -> str:
    """
    Convert text to a simple dash-separated, lowercase slug.
    Example: "Richard Herring (Bonus Episode)" -> "richard-herring-bonus-episode"
    """
    s = unicodedata.normalize("NFKD", text or "")
    # remove parentheses but keep their content separated by space
    s = s.replace("(", " ").replace(")", " ")
    # remove all characters except word chars, whitespace and hyphen
    s = re.sub(r"[^\w\s-]", "", s)
    # collapse whitespace to single dash and strip leading/trailing dashes
    s = re.sub(r"\s+", "-", s).strip("-")
    return s.lower()

# Helper function to generate the guests name from the raw ep title, needed to match episodes to restaurant mentions
# Restaurant mentions are listed by guest name, not ep number or slug
# Used in create_ep_slugs_and_guests_from_html
def extract_guest_name(raw_title: str) -> str:
    """
    Extract guest name using the simple rule:
      - split on first colon ':'
      - take the right hand side if a separator exists
      - remove any trailing parenthetical content e.g. ' (Bonus Episode)'
      - strip whitespace
    """
    if not raw_title:
        return ""

    s = raw_title.strip()

    # Split on the first recognized separator in the remaining string.
    # We prefer colon first as your original method did; then hyphens or em-dash.
    if ":" in s:
        parts = s.split(":", 1)
        candidate = parts[1].strip()
    else:
        # no separator found: either the whole string *is* the guest (as for new episodes)
        candidate = s

    # remove any parenthetical content at end or inside e.g "Name (Live) extra"
    candidate = re.sub(r"\(.*?\)", "", candidate).strip()

    # final clean: collapse multiple spaces
    candidate = re.sub(r"\s+", " ", candidate).strip()

    return candidate



# Function to create a url from new dataframe rows in episodes metadata (slugs version)
# Replaces old version of _create_url_from_row
def _create_url_from_row(row: pd.Series) -> str:
    """Creates a podscripts transcript URL from an episode's metadata."""
    slug = row["slug"]
    url = f"{transcript_base_url}{slug}"
    return url


# -------------------------------------------------------------------------
# Restaurant mentions processsing
# -------------------------------------------------------------------------
def _clean_res(res_element: BeautifulSoup) -> Tuple[str, List[str]]:
    """
    Extracts and cleans a restaurant name and its mentions from a BeautifulSoup `li` element.

    Args:
        res_element (BeautifulSoup): The BeautifulSoup Tag element for a restaurant.

    Returns:
        Tuple[str, List[str]]: The cleaned restaurant name and a list of guests who mentioned it.
    """
    text = res_element.text
    csplit = text.split("(")
    res_name = (
        re.sub(r"[^\w\s]", "", csplit[0].strip().replace("&", "and"))
        .replace("é", "e")
        .replace("ô", "o")
    )
    res_mentions = csplit[1].strip().strip(")").split(",")
    return (res_name, res_mentions)


def _create_restaurants_by_res_name_dict(html_string: str) -> Dict[str, List[str]]:
    """
    Creates a dictionary of restaurants and their mentions from the restaurants site HTML.

    Args:
        html_string (str): The HTML content of the restaurants page.

    Returns:
        Dict[str, List[str]]: Keys are restaurant names and values are a list of guests who mentioned them.
    """
    res_site_html = BeautifulSoup(html_string, features="html.parser")
    res_elements = res_site_html.find_all("li")
    restaurants_by_res_name = {}
    for item in res_elements:
        if not "(" in item.text:
            continue
        else:
            restaurants_by_res_name[_clean_res(item)[0]] = _clean_res(item)[1]
    return restaurants_by_res_name


# -------------------------------------------------------------------------
# Cleaning transcripts
# -------------------------------------------------------------------------


def _get_episode_sentences(html_string: str) -> List[BeautifulSoup]:
    """
    Finds all 'single-sentence' div elements in an HTML string.
    """
    soup = BeautifulSoup(html_string, features="html.parser")
    all_sentences = soup.find_all("div", class_="single-sentence")
    return all_sentences


def _clean_transcript_str_from_sentences(sentences: List[BeautifulSoup]) -> str:
    """
    Cleans a list of BeautifulSoup sentence tags into a single, lowercased string.
    """
    cleaned_line_list = []
    for section in sentences:
        # make lowercase
        text_lower = section.text.lower()
        # Split into lines
        lines = text_lower.splitlines()
        # Clean whitespace from end/start of lines, and don't include empty lines (recall list comp conditional at the end, and empty string is falsy)
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        # Join lines using .join, with a space between then
        single_line_text = " ".join(cleaned_lines)
        if single_line_text:
            cleaned_line_list.append(single_line_text)
    all_text_single_line = " ".join(cleaned_line_list)
    return all_text_single_line


def _clean_transcript_str_from_html(html_filepath: str) -> str:
    """
    Reads an HTML file, extracts sentences, and returns a cleaned transcript string.
    """
    html_text = try_read_html_string_from_filepath(html_filepath)
    sentences = _get_episode_sentences(html_text)
    transcript_str = _clean_transcript_str_from_sentences(sentences)
    return transcript_str


# -------------------------------------------------------------------------
# Collating timestamps
# -------------------------------------------------------------------------


def _extract_timestamps_as_list_of_dicts(
    transcript_str: str, slug: str
) -> List[Dict[str, Any]]:
    """
    Finds all 'starting point is HH:MM:SS' timestamps in a transcript string.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dict contains the episode
                               number, timestamp string, and its starting index.
    """
    timestamp_pattern = re.compile(r"starting point is (\d{2}:\d{2}:\d{2})")
    all_timestamps_in_transcript = []
    for match in timestamp_pattern.finditer(transcript_str):
        # Get the captured timestamp string (e.g., "00:00:05")
        actual_time_string = match.group(1)
        # We use group(1) because that's our (HH:MM:SS) part, group(0) refers to the whole string by default

        # Get the starting index of the entire match
        start_position_in_text = match.start()
        # Store this as a dict with episode_slug as key
        stamp_dict = {
            "slug": slug,
            "timestamp": actual_time_string,
            "start_index": start_position_in_text,
        }
        # Store this extracted data (the timestamp string and its position)
        all_timestamps_in_transcript.append(stamp_dict)
    return all_timestamps_in_transcript


# -------------------------------------------------------------------------
# Fuzzymatching and timestamp location
# -------------------------------------------------------------------------

# --- Function to find timestamp ---


def _create_list_tuple_clean_sen_og_sen_og_index(
    text: str,
) -> List[Tuple[str, str, int]]:
    """
    Takes in a clean transcript string, and creates a list of tuples containing cleaned sentences
    for fuzzymatching, original sentences and starting index for locating quotes.

    Splits text using delimiter ". ". Assumes no sentences start with puntuation (leading spaces are the only shift from the start of the original to the start
    of the cleaned sentence).

    Returns:
        List[Tuple[str, str, int]]: a list containing a tuple, with cleaned sentence, original
                                    stripped sentence, and true start index (the start index of the original sentence,
                                    in the original text).

    """
    results = []
    current_idx_in_original = 0  # This tracks our position in the original 'text'

    # Split into 'segments' (what will become sentences) by full stop/space.
    segments = text.split(". ")

    for i, segment in enumerate(
        segments
    ):  # Note enumerate is a way to loop and get index (rather than a manual counter)
        original_full_sentence_segment = segment
        # Calculate the actual start index of the content within the segment itself (after stripping leading/trailing spaces)
        # It asssumes the start index (in processes sentence) will only move due to leading spaces
        # So, it calculates the original (assuming none start with punctuation), and retains it
        # Later, we will use this original index to compare against timestamps
        leading_spaces_count = len(original_full_sentence_segment) - len(
            original_full_sentence_segment.lstrip()
        )
        true_start_index = current_idx_in_original + leading_spaces_count

        original_sentence_stripped = original_full_sentence_segment.strip()

        # Only process if the sentence is not empty after stripping
        if original_sentence_stripped:
            # Apply original cleaning, explicitly converting to lowercase for fuzzy matching
            cleaned_sentence = re.sub(
                r"[^\w\s]", "", original_sentence_stripped
            ).lower()

            # Store cleaned, original, and start index
            results.append(
                (cleaned_sentence, original_sentence_stripped, true_start_index)
            )

        # Update current_idx_in_original for the next segment.
        # Add the length of the current segment and the delimiter length (2 for ". ").
        # This assumes all segments (except possibly the last) were followed by ". ".
        current_idx_in_original += len(original_full_sentence_segment)
        if (
            i < len(segments) - 1
        ):  # Only add delimiter length if it's not the last segment
            current_idx_in_original += len(". ")

    return results


def _find_timestamp(
    original_sentence_start_index: int, transcript_timestamps: List[dict]
):
    """
    Finds the nearest timestamp occurring before or at a given sentence index.

    This function searches through a list of timestamp dictionaries (which should
    be pre-sorted by `start_index`) to find the timestamp that immediately
    precedes or is at the start of a matched sentence.

    Args:
        original_sentence_start_index (int): The starting index of the sentence
            in the full transcript string.
        transcript_timestamps (List[dict]): A list of dictionaries, where each dict
            contains 'start_index' and 'timestamp' for a periodic timestamp.

    Returns:
        Optional[str]: The timestamp string (e.g., '00:01:23') if a match is found,
                       otherwise returns None.
    """
    if original_sentence_start_index is None:
        return None
    # Could sort timestamps here for good practice, but should be sorted already
    # Reverse-iterate over timestamps to find the "nearest before or at"
    for timestamp_dict in reversed(transcript_timestamps):
        if timestamp_dict["start_index"] <= original_sentence_start_index:
            return timestamp_dict["timestamp"]

    return None  # If no timestamp found before the quote's starting position (all eps start "Starting point is 00:00:00")


def _matches_by_res_name_from_list_of_res_names(
    restaurant_names: List[str], searchable_sentences: List[str], min_score: int
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Finds fuzzy matches for a list of restaurant names within a list of cleaned sentences.

    This function iterates through each restaurant name and uses fuzzy matching to find
    sentences that are a close match. Matches are filtered based on a minimum score.

    Args:
        restaurant_names (List[str]): A list of restaurant names to search for.
        searchable_sentences (List[str]): A list of pre-cleaned sentences to search within.
        min_score (int): The minimum fuzzy match score (from 0-100) to consider
                         a match valid.

    Returns:
        Dict[str, List[Tuple[str, int, int]]]: A dictionary where:
            - Keys are the restaurant names from `restaurant_names`.
            - Values are a list of filtered matches for that restaurant.
            - Each match is a tuple containing:
                - str: The matched sentence text.
                - int: The fuzzy matching score.
                - int: The index of the matched sentence in the `searchable_sentences` list.
    """
    filtered_matches_by_string = {}
    for res_name in restaurant_names:
        matches = process.extract(
            res_name, searchable_sentences, scorer=fuzz.partial_ratio, limit=20
        )

        filtered_matches = []
        # --- FIX: Unpack the tuple of 2 items correctly ---
        for match_text, score in matches:
            if score >= min_score:
                # Find the index of the matched sentence in the original list
                # We use a try-except block for robustness in case of unexpected data.
                try:
                    original_sentence_index = searchable_sentences.index(match_text)
                    # Append all three pieces of information
                    filtered_matches.append(
                        (match_text, score, original_sentence_index)
                    )
                except ValueError:
                    # This will happen if the match text isn't found in the list,
                    # e.g., due to slight string differences not captured by .index()
                    continue

        filtered_matches_by_string[res_name] = filtered_matches

    return filtered_matches_by_string


# =========================================================================
# 4. Main Logic Functions
# =========================================================================

# -------------------------------------------------------------------------
# Epsisode metadata processsing
# -------------------------------------------------------------------------

# Function to create slugs and guest names from html 
# Replaces create_numbers_names_dict_from_html

def create_tuple_inc_ep_slugs_and_guests_list_from_html(html_string: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Parse episodes HTML and return a tuple:
      (
        [list of valid episode records],
        [list of raw_titles for excluded 'Best of' episodes]
      )
    """
    
    soup = BeautifulSoup(html_string, "html.parser")
    episode_divs = soup.find_all("div", class_="image-slide-title")

    # 1. Initialize two separate lists
    records: List[Dict[str, Any]] = []
    exceptions: List[str] = [] 

    for div in episode_divs:
        raw_title = div.get_text(separator=" ", strip=True)
        
        # 2. Check the condition using the string method
        if raw_title.startswith("Best of"):
            # 3. If it is a "Best of" episode, append the title to the exceptions list
            exceptions.append(raw_title)
            # Skip the rest of the loop for this title and move to the next 'div'
            continue
        # menus to be buried with exception?
        # christmas dinner party exception?
            
        # If the 'if' condition was false (i.e., it's a regular episode), the code continues here:
        
        guest_name = extract_guest_name(raw_title)
        slug_full = slugify(raw_title)

        records.append({
            "raw_title": raw_title,
            "slug": slug_full,
            "guest_name": guest_name
        })

    # 4. Return both lists as a tuple
    return records, exceptions


# Function to create slugs and guest names dataframe from the dict made by create_ep_slugs_and_guests_from_html
# Replaces create_numbers_names_df_from_dict

def create_slugs_guests_df_from_list_of_dict(titles_list: Dict) -> pd.DataFrame:
    """
    Takes the list of dicts of raw titles, slugs and guest names and returns a dataframe
    """
    df_episodes_metadata = pd.DataFrame(titles_list)
    return df_episodes_metadata


# Function to action creating urls for each row of slugs and guests dataframe
# Replaces create_urls_and_save_to_numbers_names_df
def create_urls_and_save_to_slugs_guests_df(
    input_dataframe: pd.DataFrame, output_filepath: str
) -> None:
    """
    Generates transcript URLs for a DataFrame of episode metadata and saves it.

    This function adds a new column 'url' to the input DataFrame by applying
    a helper function to each row. The modified DataFrame is then saved as a
    Parquet file to the specified path.

    Args:
        input_dataframe (pd.DataFrame): The DataFrame containing episode metadata
                                        with 'episode_number' and 'guest_name' columns.
        output_filepath (str): The full file path where the resulting DataFrame
                               will be saved in Parquet format.

    Returns:
        None: The function modifies the input DataFrame and saves a file to disk,
              but does not return a value.
    """
    df = input_dataframe
    df["url"] = df.apply(_create_url_from_row, axis=1)
    df.to_parquet(output_filepath)



# -------------------------------------------------------------------------
# Restaurant mentions processsing
# -------------------------------------------------------------------------


def create_mentions_by_res_name_dict(
    restaurants_html_filepath: str,
) -> Dict[str, List[str]]:
    """
    Takes restaurants html full filepath and creates a dict where keys are res names and values
    are lists of the guests who mentioned them.

    Args:
        restaurants_html_filepath: String filepath for the restaruatns html
    Returns:
        Dict[str, List[str]]: A dictionary wherekeys are res names and values are lists of the guests
        who mentioned them.
    """
    res_html_text = try_read_html_string_from_filepath(restaurants_html_filepath)
    if res_html_text:
        mentions_by_res_name_dict = _create_restaurants_by_res_name_dict(res_html_text)
        return mentions_by_res_name_dict


def create_return_exploded_res_mentions_df(
    mentions_by_res_name_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Takes in dict of res mentions (keys are res names, values lists of guests who mention) and creates and
    processes a dataframe, returning a dataframe with 1 row for each mention of a restaurant.

    Args:
        mentions_by_res_name_dict (Dict): A dictionary wherekeys are res names and values are lists of the guests
        who mentioned them.
    Returns:
        pd.DataFrame: A dataframe with 1 row for each mention of a restaurant (exploded)
    """
    # Convert dict to list, as it's easier for pandas to process
    # Each item in list is a tuple, with res name and list of guests => pandas understand each tuple is a row
    mentions_raw_data = list(mentions_by_res_name_dict.items())
    # Convert to df
    mentions_raw_df = pd.DataFrame(
        mentions_raw_data, columns=["restaurant_name", "guests_mentioned"]
    )
    # 'Explode' the dataframe, so each guest mention has their own row (the restairant name will be duplicated)
    exploded_restaurant_guest_df = mentions_raw_df.explode("guests_mentioned")
    # rename guests mentioned to guest name
    exploded_restaurant_guest_df = exploded_restaurant_guest_df.rename(
        columns={"guests_mentioned": "guest_name"}
    )
    # return df
    return exploded_restaurant_guest_df


# Function to combine the mentions dataframe with the ep metadata dataframe (new style, with slugs)
# Replaces combine_save_mentions_ep_metadata_dfs
def combine_save_mentions_and_ep_metadata_dfs(
    exploded_restaurants_guest_df: pd.DataFrame,
    ep_metadata_filepath: str,
    output_df_filepath: str,
) -> None:
    """
    Takes in exploded (one line per guest/mention) mentions/guest df, and ep metadata (numbers, names, url) dataframe
    filepath, and output filepath, and combines the dataframes. The combined dataframe is then saved as a
    Parquet file to the specified path.

    Args:
        exploded_restaurants_guest_df (pd.DataFrame): A dataframe with 1 row for each mention of a restaurant (exploded)
        ep_metadata_filepath (str): String filepath for the episode metadata dataframe
        output_df_filepath (str): String filepath for where to save the combined dataframe

    Returns:
        None: The function combines the dataframes, and saves to a parquet.
    """
    # Fetch metadata filepath
    df_episodes_metadata = try_read_parquet(ep_metadata_filepath)
    # Left merge on guest, with numbers, names, url (df_episodes_metadata)
    merged_df = pd.merge(
        df_episodes_metadata, exploded_restaurants_guest_df, on="guest_name", how="left"
    )
    # Aggregating rows so we have one row per episode, with a list of restaurant mentions
    # Note groupby creates groups based on the args (three identical in this case). as_index False means also have an index col (don't use first col as index)
    # Note .agg aggregates the data, it creates a new col called restaurants mentioned, from the col 'restaurant_name', applying the method 'dropna' to each group (restuarants that were in the restaurant_name cell), dropna gets rid of the NaN's
    # Note NaN's are placeholders for missing data (means ilterally not a number, which is confusing as it could be text...)
    ep_meta_and_mentions_df = (
        merged_df.groupby(["guest_name", "url", "slug"], as_index=False, sort=False)
        .agg(restaurants_mentioned=("restaurant_name", lambda x: list(x.dropna())))
        .rename(columns={"restaurant_name": "restaurants_mentioned"})
    )
    # Save the dataframe
    ep_meta_and_mentions_df.to_parquet(output_df_filepath, index=False)



# -------------------------------------------------------------------------
# Cleaning transcripts & Collating timestamps
# -------------------------------------------------------------------------


def extract_save_clean_text_and_periodic_timestamps(
    full_episodes_metadata_path: str, transcripts_dir: str, output_filepath: str
) -> None:
    """
    Takes the full episodes metadata filepath, the transcripts html directory, and an output filepath, and iterates
    through the episodes, processing the html into clean transcript text and collating the periodic timestamps.

    These transcripts and periodic timestamps are saved in a dataframe, which is saves as a parquet file to the
    output filepath.

    Args:
        full_episodes_metadata_path (str): The full episodes metadata dataframe filepath
        transcripts_dir (str): The directory containing the html of each episode.
        output_filepath (str): The filepath the output df is saved to.
    Returns:
        None: A dataframe containing the clean text and the timestamps (a list of Dicts) is saved to the
        output filepath as a parquet.
    """
    # 1. Load episodes meta_data
    episodes_df = try_read_parquet(full_episodes_metadata_path)
    if episodes_df is None or episodes_df.empty:
        print(
            "  ERROR: Input episode metadata is missing or empty. Cannot process transcripts."
        )
        raise ValueError("No episodes to process.")

    processed_records = []  # To store data for the final DataFrame

    # 2. Iterate through each episode's metadata
    for index, row in episodes_df.iterrows():
        episode_slug = row.get("slug")
        guest_name = row.get("guest_name")
        transcript_filename = f"{episode_slug}.html"
        transcript_filepath = os.path.join(transcripts_dir, transcript_filename)
        restaurants_mentioned = row.get("restaurants_mentioned")
        # Confirm file exists and skip if not
        if not os.path.exists(transcript_filepath):
            print(
                f"  WARNING: Transcript file not found for Episode {guest_name}, slug: {episode_slug} at {transcript_filepath}. Skipping."
            )
            continue  # Skip to the next episode
        try:
            clean_transcript_str = _clean_transcript_str_from_html(transcript_filepath)
            timestamps = _extract_timestamps_as_list_of_dicts(
                clean_transcript_str, episode_slug
            )

            processed_records.append(
                {
                    "slug": episode_slug,
                    "guest_name": guest_name,
                    "restaurants_mentioned": restaurants_mentioned,
                    "clean_transcript_text": clean_transcript_str,
                    "periodic_timestamps": timestamps,  # This will be a list of dictionaries
                }
            )
            print(
                f"  Processed Episode {episode_slug} ({guest_name}): Extracted text and {len(timestamps)} timestamps."
            )

        except Exception as e:
            print(
                f"  ERROR: Failed to process transcript for Episode {episode_slug} ({guest_name}) from {transcript_filepath}: {e}"
            )
            continue  # For MVP, just skip and warn

        if processed_records:
            result_df = pd.DataFrame(processed_records)
            result_df.to_parquet(output_filepath, index=False)
            print(
                f"Successfully saved clean transcripts and timestamps for {len(result_df)} episodes to {output_filepath}"
            )
        else:
            print(
                "No transcripts were successfully processed. Output DataFrame will be empty."
            )
            pd.DataFrame().to_parquet(output_filepath, index=False)  # Save an empty DF


# -------------------------------------------------------------------------
# Combining episode metadata with transcripts and timestamps
# -------------------------------------------------------------------------


def combine_timestamps_and_metadata(
    transcripts_timestamps_filepath: str, metadata_filepath: str
) -> pd.DataFrame:
    """
    Reads and combines the transcripts and timestamps dataframe with the metadata dataframe.

    Args:
        transcripts_timestamps_filepath(str)
        metadata_filepath (str)
    Returns:
        pd.DataFrame: A dataframe containing episode slug, restaurants mentioned, clean transcript,
        and timestamps.
    """
    metadata_df = try_read_parquet(metadata_filepath)
    transcripts_timestamps_df = try_read_parquet(transcripts_timestamps_filepath)
    combined_df = transcripts_timestamps_df.merge(
        metadata_df[["slug", "restaurants_mentioned"]],
        on="slug",
        how="left",
    )
    return combined_df


# -------------------------------------------------------------------------
# Fuzzymatching and timestamp location for easy wins (top match)
# -------------------------------------------------------------------------
def find_top_match_and_timestamps(
    combined_df: pd.DataFrame, min_match_score: int = 90
) -> pd.DataFrame:
    """
    Finds fuzzy matches for restaurant mentions in episode transcripts and associates them with timestamps.

    This function iterates through each episode's metadata and transcript data. For each mentioned
    restaurant, it performs a fuzzy search within the transcript. It then returns a DataFrame
    of the top matches and their corresponding timestamps, or notes if no match was found.

    Args:
        combined_df (pd.DataFrame): A DataFrame containing episode metadata, cleaned transcripts,
                                    and periodic timestamps.
        min_match_score (int): The minimum fuzzy match score (0-100) required to consider
                               a match valid.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a restaurant mention. It contains
                      the following columns:
                          - 'slug': The episode slug e.g. ep-217-ross-noble or elle-fanning
                          - 'Restaurant': The name of the restaurant mentioned.
                          - 'Mention text': The original sentence where the mention was found.
                          - 'Match Score': The fuzzy match score.
                          - 'Match Type': The type of match (e.g., 'full, over 90' or 'No match found').
                          - 'Timestamp': The nearest preceding timestamp for the mention.
                          - 'Transcript sample': A short sample of the transcript text.
    """
    all_mentions_collected = []

    for index, combined_row in combined_df.iterrows():
        slug = combined_row.get("slug")
        guest_name = combined_row.get("guest_name")
        clean_transcript_text = combined_row.get("clean_transcript_text")
        periodic_timestamps = combined_row.get("periodic_timestamps")

        restaurants_data = combined_row.get("restaurants_mentioned", [])
        transcript_sample = (
            clean_transcript_text[:200]
            if isinstance(clean_transcript_text, str)
            else "No Transcript Found"
        )

        # Unsure what data type the res mentions are, hence need for this
        restaurants_list = []
        if isinstance(restaurants_data, list):
            restaurants_list = restaurants_data
        elif isinstance(restaurants_data, np.ndarray) and restaurants_data.size > 0:
            # Flatten the array and convert it to a standard Python list of strings
            restaurants_raw_list = restaurants_data.flatten().tolist()
            restaurants_list = [
                name.strip().lower() for name in restaurants_raw_list if name.strip()
            ]
        elif isinstance(restaurants_data, str):
            restaurants_list = [
                name.strip() for name in restaurants_data.split(",") if name.strip()
            ]

        if restaurants_list:
            episode_sentences_data = _create_list_tuple_clean_sen_og_sen_og_index(
                clean_transcript_text
            )
            searchable_sentences = [
                item[0] for item in episode_sentences_data
            ]  # This is to select the cleaned sentence from the list of tuple
            # of cleaned sentence, original, and true start index that create_sentence_list creates

            all_matches_for_episode = _matches_by_res_name_from_list_of_res_names(
                restaurants_list, searchable_sentences, 90
            )
            # --- all_matches_for_episode is a dict with key res_name and value lists of matches (matches r tuples of quote, score)
            for (
                restaurant_name_query,
                match_list_for_query,
            ) in all_matches_for_episode.items():
                if match_list_for_query:
                    top_match = match_list_for_query[0]
                    # Unpack the top match's data
                    matched_cleaned_text, score, matched_sentence_index = top_match
                    original_sentence_data = episode_sentences_data[
                        matched_sentence_index
                    ]  # This takes you back to episode sentences data for the sentence index
                    # Which is a tuple of clean sentence, original, and index of sentence within sen list
                    original_sentence_text = original_sentence_data[
                        1
                    ]  # The og sentence is at index 1 in this tuple
                    original_start_index = original_sentence_data[
                        2
                    ]  # The og start index is at index 2 in this tuple

                    timestamp = _find_timestamp(
                        original_start_index, periodic_timestamps
                    )

                    mention = {
                        "Episode ID": slug,
                        "Restaurant": restaurant_name_query,
                        "Mention text": original_sentence_text,
                        "Match Score": score,
                        "Match Type": f"full, over {min_match_score}",
                        "Timestamp": timestamp,
                        "transcript_sample": transcript_sample,
                    }
                    all_mentions_collected.append(mention)
                else:
                    null_mention = {
                        "Episode ID": slug,
                        "Restaurant": restaurant_name_query,
                        "Mention text": None,
                        "Match Score": 0,
                        "Match Type": "No match found",
                        "Timestamp": None,
                        "transcript_sample": transcript_sample,
                    }
                    all_mentions_collected.append(null_mention)
        else:
            print(
                f"  No raw mentions found in 'restaurants_mentioned' list for Episode {slug}. Skipping"
            )
    combined_df = pd.DataFrame(all_mentions_collected)
    return combined_df


# =========================================================================
# 5. Script exectuion
# This section contains script that runs only when this script is run directly when it is open (not when called by another script)
# This will contain a smaller model of the processes, so we can test before implementing in main
# N.B. OUTDATED
# =========================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Epsisode metadata processsing
    # -------------------------------------------------------------------------

    # Episode slugs and guest names extraction
    print("=== Testing episode slugs and guests extraction ===")

    # Small test HTML snippet or file path
    episodes_html_filepath = os.path.join("data/test_temp", "episodes.html")
    try:
        with open(episodes_html_filepath, "r", encoding="utf-8") as f:
            html_text = f.read()

        # Create slug -> guest dict
        slugs_guests_and_exclusions_tuple = create_tuple_inc_ep_slugs_and_guests_list_from_html(html_text)
        slugs_guests_dict_list = slugs_guests_and_exclusions_tuple[0]
        print("Sample slugs -> guest mapping:", list(slugs_guests_dict_list[:3]))

        # Convert to DataFrame
        df_episodes_metadata = create_slugs_guests_df_from_list_of_dict(slugs_guests_dict_list)
        print("Episode metadata DF head:")
        print(df_episodes_metadata.head())

    except FileNotFoundError:
        print(f"Error: {episodes_html_filepath} not found.")


    # ---Testing function to create URL's using dummy data---

    print("\n=== Testing URL creation ===")
    df_episodes_metadata_with_urls_filepath = os.path.join(test_temp_dir, "df_episodes_metadata_with_urls.parquet")
    create_urls_and_save_to_slugs_guests_df(df_episodes_metadata, df_episodes_metadata_with_urls_filepath)
    df_episodes_metadata_with_urls = try_read_parquet(df_episodes_metadata_with_urls_filepath)
    print("URLs added to DF head:")
    print(df_episodes_metadata_with_urls.head())

    # -------------------------------------------------------------------------
    # Restaurant mentions processsing
    # -------------------------------------------------------------------------
    print("\n=== Testing restaurant mentions processing ===")
    # ---Testing creating the restaurant mentions dict, converting into dataframe---

    # Collecting the html, creating dict
    res_html_test_filepath = os.path.join(
        project_root, "data/test_temp/restaurants_site.html"
    )
    test_res_html_text = try_read_html_string_from_filepath(res_html_test_filepath)
    if test_res_html_text:
        restaurants_by_res_name_dict = _create_restaurants_by_res_name_dict(
            test_res_html_text
        )
    print("\n===First ten restaurants and guests who mention them: ===")
    print(list(restaurants_by_res_name_dict.items())[:10])

    # 1. Create the dataframe, with guest mentioned still as a list
    # Note first conver dict to list, as it's easier for pandas to then have a col with a list of names in

    mentions_raw_data = list(restaurants_by_res_name_dict.items())
    # 2. Now, create the DataFrame directly from this list of tuples.
    # Pandas understands each tuple is a row, and we can assign the column names directly.
    mentions_raw_df = pd.DataFrame(
        mentions_raw_data, columns=["restaurant_name", "guests_mentioned"]
    )
    print("\n=== Corrected mentions DataFrame (first 5 rows):\n", mentions_raw_df.head())

    # 3. 'Explode' the dataframe, so each guest mention has their own row (the restairant name will be dupicated)
    restaurant_guest_df = mentions_raw_df.explode("guests_mentioned")

    # 4. After exploding, the column with a single guest is still named 'guests_mentioned'.
    # We rename it to 'guest_name' to match the column in your episodes_df.
    restaurant_guest_df = restaurant_guest_df.rename(
        columns={"guests_mentioned": "guest_name"}
    )
    print("\n===Res mentions, numbers and names collated dataframe test")
    print(restaurant_guest_df.head())

    # Saving dataframe so I can combine with episodes data
    test_processed_filepath = os.path.join(
        test_temp_dir, "restaurant_mentions_test.parquet"
    )
    restaurant_guest_df.to_parquet(test_processed_filepath, index=False)

    # -------------------------------------------------------------------------
    # Collating restaurant mentions and episodes metadata
    # -------------------------------------------------------------------------
    # ---Testing merging restaurant mentions dataframe with numbers, names, url---

    print("\n=== Testing combining episode metadata and mentions ===")
    test_full_episodes_metadata_combined_path = os.path.join(test_temp_dir, "test_full_episodes_metadata_slugs_combined.parquet")
    print(f"restaurant_guest_df type is {type(restaurant_guest_df)}")
    print(f"df_episodes_metadata_with_urls type is {type(df_episodes_metadata_with_urls_filepath)}")
    combine_save_mentions_and_ep_metadata_dfs(
        restaurant_guest_df, df_episodes_metadata_with_urls_filepath, test_full_episodes_metadata_combined_path
    )
    combined_df_head = try_read_parquet(test_full_episodes_metadata_combined_path).head()
    print("\n=== Combined df head ===")
    print(combined_df_head.head())

    # Saving combined df (metadata, res mentions) to filepath for use in timestamp collation function
    combined_df_head_filepath = os.path.join(test_temp_dir, "combined_df_head")
    combined_df_head.to_parquet(combined_df_head_filepath)
    

    # -------------------------------------------------------------------------
    # Downloading transcripts
    # -------------------------------------------------------------------------

    # Testing transcript productinon
    print("\n=== Testing transcript download ===")
    transcripts_dir = os.path.join("data/test_temp", "transcripts_sample")
    output_filepath = os.path.join("data/test_temp", "test_transcripts_data_processing_script.parquet")

    orchestrate_scraper(
        df=combined_df_head,  # only a small batch for testing
        base_url= transcript_base_url,
        out_dir=transcripts_dir,
        max_attempts_per_url=3,
        max_workers=1)
    # -------------------------------------------------------------------------
    # Cleaning and Collating timestamps from transcripts 
    # -------------------------------------------------------------------------
    print("\n=== Testing clean transcript, timestamp production and saving ===")
    test_five_full_data_and_transcripts_timestamps_path = os.path.join(
        test_temp_dir, "test_five_full_data_and_transcripts_timestamps_slug_ver_df.parquet")
    
    try:
        extract_save_clean_text_and_periodic_timestamps(combined_df_head_filepath, transcripts_dir, test_five_full_data_and_transcripts_timestamps_path)
        print(f"\nSuccessfully extracted clean text and timestamps, here's dataframe which was saved")
        print(try_read_parquet(test_five_full_data_and_transcripts_timestamps_path))
    except Exception as e:
        print(f"\nError with extracting clean text and timestamps from combined metadata mentions df: {e}")
    

    # -------------------------------------------------------------------------
    # Fuzzywuzzy easy wins and timestamp matching
    # -------------------------------------------------------------------------

    # --- Read test dataframes, merge for ease of processing ---
    combined_test_df = try_read_parquet(test_five_full_data_and_transcripts_timestamps_path)

    # --- Run top matches on the test data ---
    top_mentions_df = find_top_match_and_timestamps(combined_test_df, 90)

    # --- Convert list into dataframe, print output ---

    print(f"\n--- TOP COLLECTED ---")
    print(f"Top Mentions DataFrame created with {len(top_mentions_df)} rows.")

    # --- TEMPORARILY CHANGE PANDAS DISPLAY SETTINGS ---
    # Set the maximum column width to a high number (or 0 for unlimited)
    pd.set_option('display.max_colwidth', None) 
    # Set max rows to ensure all head rows are displayed
    pd.set_option('display.max_rows', 500) 

    # Now print the full content
    print(top_mentions_df[['Episode ID', 'Restaurant', 'Mention text', 'Timestamp']].head(10)) 

    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.max_rows')
