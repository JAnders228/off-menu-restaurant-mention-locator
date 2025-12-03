# --- Streamlit app to display the restaurants by region dataframe ---

import streamlit as st
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent

relative_path = Path("..") / "data" / "processed" / "top_mentions_expanded_with_regions.csv"

FINAL_PATH = (script_dir / relative_path).resolve()

st.text(str(FINAL_PATH))

df = pd.read_csv(FINAL_PATH)

st.write(df)

# Original top mentions dataframe

top_mentions_relative_path = Path("..") / "data" / "processed" / "expanded_easy_win_mention_search_df.parquet"

top_mentions_final_path = (script_dir / top_mentions_relative_path).resolve()

st.text(str(top_mentions_final_path))

top_mentions_original_df = pd.read_parquet(top_mentions_final_path)

st.write(top_mentions_original_df)