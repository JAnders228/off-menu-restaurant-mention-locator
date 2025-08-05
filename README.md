# Off Menu Podcast – Restaurant Mention Locator

This project extracts the restaurants mentioned in the Off Menu Podcast, and and identifies the **first timestamp** where each one occurs.

Mentions comes from the 'restaurants' section of the website: https://www.offmenupodcast.co.uk/restaurants. 

The transcripts (with timestamps) come from podscripts: https://podscripts.co/podcasts/off-menu-with-ed-gamble-and-james-acaster/


## Key Features

- Parses transcript data from podcast HTML pages
- Matches restaurant names using fuzzy matching
- Returns timestamp of **first mention per restaurant**
- Outputs a cleaned Parquet file containing restaurant matches, timestamps, and context:
    - 'Episode ID': The episode number.
    - 'Restaurant': The name of the restaurant mentioned.
    - 'Mention text': The original sentence where the mention was found.
    - 'Match Score': The fuzzy match score.
    - 'Match Type': The type of match (e.g., 'full, over 90' or 'No match found').
    - 'Timestamp': The nearest preceding timestamp for the mention.
    - 'Transcript sample': A short sample of the transcript text.

## Project Structure

- **data/**: Directory for all data, both raw and processed.
    - **processed/**: Processed data including the results of the pipeline
    - **raw/**: Raw, untouched input data.
- **notebooks/**: Directory for Jupyter notebooks.
    - **data_extraction/**: Notebooks related to the data extraction process.
    - **data_processing/**: Notebooks for data cleaning and transformation.
    - **Analysis/**: Notebooks for analysis of results.
- **off_menu/**: The main source code directory.
    - `__init__.py`: Initializes the directory as a Python package.
    - `config.py`: Configuration settings for the project.
    - `data_extraction.py`: Scripts for data extraction.
    - `data_processing.py`: Scripts for data processing and cleaning.
    - `utils.py`: A collection of helper functions.
- `main.py`: The entry point for the main data pipeline.
- `README.md`: The project's documentation file.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `requirements.txt`: Lists all project dependencies.

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
python main.py
```

Output will be saved to data\processed\easy_win_mention_search_df.parquet

You can load it with the following command:
```Python
import pandas as pd
df = pd.read_parquet('data/processed/easy_win_mention_search_df.parquet')
```

## Future work

 - Do a second run of 'harder win' mentions, using tokenised mentions (effective in testing)
 - Utilise the transcripts on the official website to improve matching (available for many, but not all episodes)
 - Collate dream menus using fuzzymatching to identify the keywords/phrases that come before it's read
 - Build a web interface for the mentions, timestamps, dream menus
 - Perform analytics on the dream menus
 - Use an LLM to refine clean up mismatches, and possibly extract more fun text based insights (e.g. things James hates)

 ## Author and notes

 Created by J Andersen as part of a data science portfolio. See accompanying blog post (coming soon) for a breakdown of technical choices and lessons learned.