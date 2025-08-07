# Utils.py
"""
This module contains general purpose utilities (not specific to extraction or processing)
"""

# =========================================================================
# 1. Imports
# =========================================================================
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import os
from typing import List, Tuple, Dict, Any, Optional


# =========================================================================
# 2. Functions
# =========================================================================


def save_parquet_index_false(dataframe, output_dir):
    dataframe.to_parquet(output_dir, index=False)


def try_read_parquet(filepath: str) -> pd.DataFrame | None:
    """
    Safely reads a DataFrame from a Parquet file, handling errors gracefully.

    This function wraps pd.read_parquet() in a try-except block to prevent the
    program from crashing if the file is not found or corrupted.

    Args:
        filepath (str): The full path to the Parquet file to be read.

    Returns:
        pd.DataFrame | None: The DataFrame loaded from the file on success,
                             or None if an error occurs.
    """
    try:
        df = pd.read_parquet(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}. Did it save correctly?")
        return None
    except Exception as e:
        # Catch any other unexpected errors during reading
        print(f"An unexpected error occurred while reading '{filepath}':")
        print(f"Details: {e}")
        return None


def try_read_html_string_from_filepath(full_filepath) -> str | None:
    """
    Function takes in a full filepath , and returns html_text(str) if there
    is an html file there, None otherwise.

    Args:
        full filepath (str): String of the full filepath to read from

    Returns:
        str | None: returns html_text(str) if there is an html file there. Otherwise,
        prints Error and returns None
    """
    try:
        with open(full_filepath, "r", encoding="utf-8") as html:
            html_text = html.read()
            return html_text
    except FileNotFoundError:
        print(
            f"Error: The file was not found at {full_filepath}. Did it save correctly?"
        )
        return None


def extract_html(url: str) -> str | None:
    """
    Returns the HTML content as a string for the given URL, or None if the download fails.

    This function attempts to download HTML from a URL, handling common HTTP errors
    (like 404) and network issues (like timeouts). It returns the content on
    success and None on failure.

    Args:
        url (str): The URL in string form of the site to extract the HTML from.

    Returns:
        str | None: The HTML content as a string if the request is successful,
                    or None if a `requests` exception or other error occurs.
    """
    try:
        # Added a timeout to prevent indefinite waiting
        response = requests.get(url, timeout=10)
        # This will raise an HTTPError for 4xx/5xx responses (e.g., 404 Not Found)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        # This catches ALL requests-related errors:
        # - HTTPError (for 404, 500 etc. after raise_for_status)
        # - ConnectionError (no internet, host unreachable)
        # - Timeout (request took too long)
        print(f"  WARNING: Failed to extract HTML from {url}: {e}")
        return None  # Return None to indicate failure
    except Exception as e:
        # Catch any other unexpected errors
        print(f"  WARNING: An unexpected error occurred for {url}: {e}")
        return None


def save_text_to_file(content: str, filename: str, directory: str) -> None:
    """
    Saves string content (as a string) to a specified file within a given directory.

    Args:
        content (str): The HTML content as a string (e.g., from extract_html.text).
        filename (str): The name of the file to save, including its extension (e.g., "episode_1.html").
        directory (str): The path to the directory where the file should be saved
                         (e.g., "data/raw").

    Returns:
        None: The function saves the text to the specified directory
    """
    full_filepath = os.path.join(directory, filename)  # Constructs the filepath
    os.makedirs(directory, exist_ok=True)  # ensures directory exists, creates it if not
    with open(full_filepath, "w", encoding="utf-8") as file:
        file.write(content)


def num_check(text) -> bool:
    """
    This function checks if a string contains digits, this can be used to determine if an episode has a valid number
    or not (e.g. if it's a special episode).

    Args:
        text (str): The text you want to check

    Returns:
        bool:
            - False if the argument is not text, or if it doesn't meet criteria below
            - True if the arg is text and the second character is a digit, or if the first char is '-' and second digit
    """
    if not text:
        return False
    if text[0] == "-":
        return text[1:].isdigit()
    return text.isdigit()
