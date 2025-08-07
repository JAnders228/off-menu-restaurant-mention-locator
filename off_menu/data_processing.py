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

from .utils import try_read_html_string_from_filepath, try_read_parquet
from .config import transcript_base_url

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


def _num_check(text: str) -> bool:
    """Checks if a string represents an integer, handling positive and negative numbers."""
    if not text:
        return False
    if text[0] == "-":
        return text[1:].isdigit()
    return text.isdigit()


#  Helper function to identify the index after the first number ends, as some eps don't have the colon splitting the title and number
def _find_num_end(tag: BeautifulSoup) -> Optional[int]:
    """
    Finds the index where a number ends in a BeautifulSoup Tag's text.

    Args:
        tag (BeautifulSoup): The BeautifulSoup Tag element to scan.

    Returns:
        Optional[int]: The index after the number ends, or None if not found.
    """
    text = tag.text
    counter = 0
    while counter < len(text) - 1:
        if text[counter].isdigit() and not text[counter + 1].isdigit():
            return counter + 1
        else:
            counter += 1
            continue


def _name_num_split(episode_tag: BeautifulSoup) -> Optional[Tuple[str, str]]:
    """
    Splits an episode's HTML tag into its name and number.

    Args:
        episode_tag (BeautifulSoup): The episode's BeautifulSoup Tag element.

    Returns:
        Optional[Tuple[str, str]]: A tuple containing the episode name and number,
                                   or None if the episode is not in a standard format.
    """
    split = []
    text = episode_tag.text
    break_point = _find_num_end(episode_tag)
    split.append(text[:break_point])
    split.append(text[break_point:])
    #  The following splits the name before the number (ep number) by space, and then selects the number
    #  It only works for regular episodes
    number = split[0].split()[1]
    # Deal with "best of" episodes or episodes without numbers (not included)
    if _num_check(number) == False:
        return "not in standard form"
    # Format name - just the slice beyond the break unless there are "("
    if "(" in split[1]:
        name = split[1].split("(")[0].strip(":").strip()
        return (name, number)
    else:
        name = split[1].strip(":").strip()
        return (name, number)


def _create_epnumber_epname_dict(html_string: str) -> Tuple[Dict[int, str], List[Any]]:
    """
    Extracts episode numbers and names from the Off Menu episodes site HTML.

    Args:
        html_string (str): The HTML content of the Off Menu episodes page.

    Returns:
        Tuple[Dict[int, str], List[Any]]: A tuple containing:
            - A dictionary where keys are episode numbers (int) and values are episode names (str).
            - A list of episodes that were not included (e.g., special episodes).
    """
    episodes_site_html = BeautifulSoup(html_string, features="html.parser")
    episode_elements = episodes_site_html.find_all("div", class_="image-slide-title")
    numbers_and_names = {}
    not_included = []
    counter = 0
    # Loop through the items
    for item in episode_elements:
        name = _name_num_split(item)[0]
        number = _name_num_split(item)[1]
        #  Deal with non standard episodes
        if _num_check(number) == False:
            not_included.append(counter)
            counter += 1
            continue
        else:
            numbers_and_names[int(number)] = name
    return numbers_and_names, not_included


def _create_url_from_row(row: pd.Series) -> str:
    """Creates a podscripts transcript URL from an episode's metadata."""
    num = row["episode_number"]
    first_name = row["guest_name"].split()[0].lower()
    if len(row["guest_name"].split()) > 1:
        second_name = row["guest_name"].split()[1].lower()
    else:
        second_name = ""
    url = f"{transcript_base_url}ep-{num}-{first_name}-{second_name}"
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
    transcript_str: str, ep_num: int
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
        # Store this as a dict with episode_number as key
        stamp_dict = {
            "episode_number": ep_num,
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


def create_numbers_names_dict_from_html(episodes_html_filepath: str) -> Dict[int, str]:
    """Taken the html filepath of the episodes site and collates a dictionary of numbers
    and names.

    Args:
        episodes_html_filepath (str): the filepath for the episodes site html
    Returns:
        Dict[int, str]: A dictionary where the keys are numbers and the values names of the episodes
    """
    html_str = try_read_html_string_from_filepath(episodes_html_filepath)
    numbers_names_dict = _create_epnumber_epname_dict(html_str)[
        0
    ]  # _create_epnumber_epname_dict returns a tuple (not inc list too), hence [0]
    return numbers_names_dict


def create_numbers_names_df_from_dict(numbers_names_dict: Dict) -> pd.DataFrame:
    """
    Takes the dict of numbers and names and returns a dataframe
    """
    df_episodes_metadata = pd.DataFrame.from_dict(
        numbers_names_dict,
        orient="index",  # means dict keys become "index" col in dataframe
        columns=["guest_name"],  # means dict values
    )
    # Give the index a meaningful name
    df_episodes_metadata.index.name = "episode_number"
    # Reset the index to make 'episode_number' a regular column (instead of the index col)
    # Note could be done in one step but less readable and pandas-idiomatic
    df_episodes_metadata.reset_index(inplace=True)
    return df_episodes_metadata


def create_urls_and_save_to_numbers_names_df(
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


def combine_save_mentions_ep_metadata_dfs(
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
        merged_df.groupby(["episode_number", "guest_name", "url"], as_index=False)
        .agg(restaurants_mentioned=("restaurant_name", lambda x: list(x.dropna())))
        .rename(columns={"restaurant_name": "restaurants_mentioned"})
    )
    # Save the dataframe
    ep_meta_and_mentions_df.to_parquet(output_df_filepath, index=False)


# -------------------------------------------------------------------------
# Cleaning transcripts & Collating timestamps
# -------------------------------------------------------------------------


def extract_clean_text_and_periodic_timestamps(
    full_episodes_metadata_path: str, transcripts_dir: str, output_filepath: str
) -> None:
    """
    Takes the full episodes metadata filepath, the transcripts html directory, and an output filepath, and iterates
    through the epusides, processing the html into clean transcript text and collating the periodic timestamps.

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
        episode_number = row.get("episode_number")
        guest_name = row.get("guest_name")
        transcript_filename = f"ep_{episode_number}.html"
        transcript_filepath = os.path.join(transcripts_dir, transcript_filename)

        # Confirm file exists and skip if not
        if not os.path.exists(transcript_filepath):
            print(
                f"  WARNING: Transcript file not found for Episode {episode_number} ({guest_name}) at {transcript_filepath}. Skipping."
            )
            continue  # Skip to the next episode
        try:
            clean_transcript_str = _clean_transcript_str_from_html(transcript_filepath)
            timestamps = _extract_timestamps_as_list_of_dicts(
                clean_transcript_str, episode_number
            )

            processed_records.append(
                {
                    "episode_number": episode_number,
                    "guest_name": guest_name,
                    "clean_transcript_text": clean_transcript_str,
                    "periodic_timestamps": timestamps,  # This will be a list of dictionaries
                }
            )
            print(
                f"  Processed Episode {episode_number} ({guest_name}): Extracted text and {len(timestamps)} timestamps."
            )

        except Exception as e:
            print(
                f"  ERROR: Failed to process transcript for Episode {episode_number} ({guest_name}) from {transcript_filepath}: {e}"
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
        pd.DataFrame: A dataframe containing episode number, restaurants mentioned, clean transcript,
        and timestamps.
    """
    metadata_df = try_read_parquet(metadata_filepath)
    transcripts_timestamps_df = try_read_parquet(transcripts_timestamps_filepath)
    combined_df = transcripts_timestamps_df.merge(
        metadata_df[["episode_number", "restaurants_mentioned"]],
        on="episode_number",
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
                          - 'Episode ID': The episode number.
                          - 'Restaurant': The name of the restaurant mentioned.
                          - 'Mention text': The original sentence where the mention was found.
                          - 'Match Score': The fuzzy match score.
                          - 'Match Type': The type of match (e.g., 'full, over 90' or 'No match found').
                          - 'Timestamp': The nearest preceding timestamp for the mention.
                          - 'Transcript sample': A short sample of the transcript text.
    """
    all_mentions_collected = []

    for index, combined_row in combined_df.iterrows():
        episode_number = combined_row.get("episode_number")
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
                        "Episode ID": episode_number,
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
                        "Episode ID": episode_number,
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
                f"  No raw mentions found in 'restaurants_mentioned' list for Episode {episode_number}. Skipping"
            )
    combined_df = pd.DataFrame(all_mentions_collected)
    return combined_df


# =========================================================================
# 5. Script exectuion
# This section contains script that runs only when this script is run directly when it is open (not when called by another script)
# This will contain a smaller model of the processes, so we can test before implementing in main
# =========================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Epsisode metadata processsing
    # -------------------------------------------------------------------------

    # ---Testing function to create numbers / names dict---
    print(f"Episodes html filepath: {episodes_html_filepath}")
    try:
        with open(episodes_html_filepath, "r", encoding="utf-8") as html:
            html_text = html.read()
        numbers_names_dict = _create_epnumber_epname_dict(html_text)[0]
        print(f"First episode name: {numbers_names_dict[1]}")
    except FileNotFoundError:
        print(
            f"Error: The file was not found at {episodes_html_filepath}. Did it save correctly?"
        )

    # ---Testing creating the dataframe of number / names (using full dict as quick)---

    print(f"DEBUG: Type of numbers_names_dict: {type(numbers_names_dict)}")

    # Create the dataframe, with dict keys as index, ep names as column
    df_episodes_metadata = pd.DataFrame.from_dict(
        numbers_names_dict,
        orient="index",  # means dict keys become "index" col in dataframe
        columns=["guest_name"],  # means dict values
    )
    # Give the index a meaningful name
    df_episodes_metadata.index.name = "episode_number"
    # Reset the index to make 'episode_number' a regular column (instead of the index col)
    # Note could be done in one step but less readable and pandas-idiomatic
    df_episodes_metadata.reset_index(inplace=True)
    # Print head and info
    print("DataFrame created successfully. Here's its head:")
    print(df_episodes_metadata.head())
    print("\nDataFrame information:")
    df_episodes_metadata.info()  # Gives summary of columns and data types

    # ---Testing function to create URL's using dummy data---

    data = {
        "episode_number": [1, 2, 294],
        "guest_name": ["James Acaster", "Ed Gamble", "Carey Mulligan"],
    }
    df = pd.DataFrame(data)

    # Apply the function over each row of the df
    df["url"] = df.apply(_create_url_from_row, axis=1)

    # Print test output
    print("Test output from adding URL to numbers and names df")
    print(df.info())

    # ---Testing applying URL function to numbers and names df?? Or does this belong in main...---

    # Dummy data (head of the numbers and names dataframe)
    # Note ep_meta_and_mentions_head already contains epnum, guestname, url, mentions
    # NOT THE CORRECT DF TO BE TESTING THIS ON
    # Apply the function
    df_episodes_metadata["url"] = df_episodes_metadata.apply(
        _create_url_from_row, axis=1
    )
    # Print output
    print("\nTest adding URL to numbers and names")
    print(df_episodes_metadata)

    # -------------------------------------------------------------------------
    # Restaurant mentions processsing
    # -------------------------------------------------------------------------

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
    print(list(restaurants_by_res_name_dict.items())[:10])

    # 1. Create the dataframe, with guest mentioned still as a list
    # Note first conver dict to list, as it's easier for pandas to then have a col with a list of names in

    mentions_raw_data = list(restaurants_by_res_name_dict.items())
    # 2. Now, create the DataFrame directly from this list of tuples.
    # Pandas understands each tuple is a row, and we can assign the column names directly.
    mentions_raw_df = pd.DataFrame(
        mentions_raw_data, columns=["restaurant_name", "guests_mentioned"]
    )
    print("Corrected mentions DataFrame (first 5 rows):\n", mentions_raw_df.head())

    # 3. 'Explode' the dataframe, so each guest mention has their own row (the restairant name will be dupicated)
    restaurant_guest_df = mentions_raw_df.explode("guests_mentioned")

    # 4. After exploding, the column with a single guest is still named 'guests_mentioned'.
    # We rename it to 'guest_name' to match the column in your episodes_df.
    restaurant_guest_df = restaurant_guest_df.rename(
        columns={"guests_mentioned": "guest_name"}
    )
    print("Res mentions, numbers and names collated dataframe test")
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

    # Fetch restaurant/guest dataframe
    filepath = os.path.join(test_temp_dir, "restaurant_mentions_test.parquet")
    restaurant_mentions_test_df = try_read_parquet(filepath)
    # Left merge on guest, with numbers, names, url (df_episodes_metadata)
    merged_df = pd.merge(
        df_episodes_metadata, restaurant_mentions_test_df, on="guest_name", how="left"
    )
    print("---Merged df head---")
    print(merged_df.head())
    # Aggregating rows so we have one row per episode, with a list of restaurant mentions
    # Note groupby creates groups based on the args (three identical in this case). as_index False means also have an index col (don't use first col as index)
    # Note .agg aggregates the data, it creates a new col called restaurants mentioned, from the col 'restaurant_name', applying the method 'dropna' to each group (restuarants that were in the restaurant_name cell), dropna gets rid of the NaN's
    # Note NaN's are placeholders for missing data (means ilterally not a number, which is confusing as it could be text...)

    ep_meta_and_mentions_df = (
        merged_df.groupby(["episode_number", "guest_name", "url"], as_index=False)
        .agg(restaurants_mentioned=("restaurant_name", lambda x: list(x.dropna())))
        .rename(columns={"restaurant_name": "restaurants_mentioned"})
    )

    print("---Final (ep_meta_and_mentions) dataframe head---")
    print(ep_meta_and_mentions_df.head())

    # Saving the dataframe
    test_processed_filepath = os.path.join(
        test_temp_dir, "ep_meta_and_mentions.parquet"
    )
    ep_meta_and_mentions_df.to_parquet(test_processed_filepath, index=False)

    # Saving just the head for testing purposes
    head_filepath = os.path.join(test_temp_dir, "ep_meta_and_mentions_head.parquet")
    ep_meta_and_mentions_head = ep_meta_and_mentions_df.head()
    ep_meta_and_mentions_head.to_parquet(head_filepath, index=False)

    # -------------------------------------------------------------------------
    # Cleaning transcripts
    # -------------------------------------------------------------------------

    # ---Step 1: Test producing clean string transcript from html---

    # Testing transcript productinon
    test_filepath = os.path.join(project_root, "data/test_temp/ep_1.html")

    def test_transcript_str(transcript_html_filepath):
        try:
            with open(test_filepath, "r", encoding="utf-8") as html:
                html_text = html.read()
            transcript_str = _clean_transcript_str_from_sentences(
                _get_episode_sentences(html_text)
            )
            print(transcript_str[:100])
            return transcript_str
        except FileNotFoundError:
            print(
                f"Error: The file was not found at {test_filepath}. Did it save correctly?"
            )

    # -------------------------------------------------------------------------
    # Cleaning and Collating timestamps function
    # -------------------------------------------------------------------------

    def test_extract_clean_text_and_periodic_timestamps(
        full_episodes_metadata_path, transcripts_dir, output_filepath
    ):
        # 1. Load episodes meta_data
        episodes_df = try_read_parquet(full_episodes_metadata_path)
        if episodes_df is None or episodes_df.empty:
            print(
                "  ERROR: Input episode metadata is missing or empty. Cannot process transcripts."
            )
            raise ValueError("No episodes to process.")

        processed_records = []  # To store data ready for producing the final DataFrame

        # 2. Iterate through each episode's metadata
        for index, row in episodes_df.head().iterrows():
            episode_number = row.get("episode_number")
            guest_name = row.get("guest_name")
            transcript_filename = f"ep_{episode_number}.html"
            transcript_filepath = os.path.join(transcripts_dir, transcript_filename)

            # 3. Confirm file exists and skip if not
            if not os.path.exists(transcript_filepath):
                print(
                    f"  WARNING: Transcript file not found for Episode {episode_number} ({guest_name}) at {transcript_filepath}. Skipping."
                )
                continue  # Skip to the next episode
            # 4. cleaning and timestamp collation
            try:
                clean_transcript_str = _clean_transcript_str_from_html(
                    transcript_filepath
                )
                timestamps = _extract_timestamps_as_list_of_dicts(
                    clean_transcript_str, episode_number
                )
                # 5. Adding to processed records
                processed_records.append(
                    {
                        "episode_number": episode_number,
                        "guest_name": guest_name,
                        "clean_transcript_text": clean_transcript_str,
                        "periodic_timestamps": timestamps,  # This will be a list of dictionaries
                    }
                )
                print(
                    f"  Processed Episode {episode_number} ({guest_name}): Extracted text and {len(timestamps)} timestamps."
                )

            except Exception as e:
                print(
                    f"  ERROR: Failed to process transcript for Episode {episode_number} ({guest_name}) from {transcript_filepath}: {e}"
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

    # Testing:
    full_episodes_metadata_path = os.path.join(
        test_temp_dir, "ep_meta_and_mentions.parquet"
    )
    transcripts_dir = test_temp_dir
    output_filepath = os.path.join(
        test_temp_dir, "test_clean_text_and__timestamps_df.parquet"
    )

    test_extract_clean_text_and_periodic_timestamps(
        full_episodes_metadata_path, transcripts_dir, output_filepath
    )

    # -------------------------------------------------------------------------
    # Fuzzywuzzy easy wins and timestamp matching
    # -------------------------------------------------------------------------

    # --- Read test dataframes, merge for ease of processing ---

    first_five_transcripts_timestamps_path = os.path.join(
        test_temp_dir, "test_clean_text_and__timestamps_df.parquet"
    )
    meta_head_filepath = os.path.join(
        test_temp_dir, "ep_meta_and_mentions_head.parquet"
    )
    combined_test_df = combine_timestamps_and_metadata(
        first_five_transcripts_timestamps_path, meta_head_filepath
    )

    # --- Run top matches on the test data ---
    print()
    top_mentions_df = find_top_match_and_timestamps(combined_test_df, 90)

    # --- Convert list into dataframe, print output ---

    print(f"\n--- TOP COLLECTED ---")
    print(f"Top Mentions DataFrame created with {len(top_mentions_df)} rows.")
    print(top_mentions_df)
