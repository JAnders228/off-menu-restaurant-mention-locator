# app.py (updated to prefer expanded & highlighted mention text)
import streamlit as st
import pandas as pd
import re
from pathlib import Path
from typing import Optional
import html

# ---------- Settings ----------
script_dir = Path(__file__).resolve().parent
relative_path = Path("..") / "data" / "processed" / "top_mentions_expanded_with_regions.csv"
FINAL_PATH = (script_dir / relative_path).resolve()
CSV_PATH = str(FINAL_PATH)

# ---------- Config (colors) ----------
OFFMENU_BG = "#f7f4ef"
OFFMENU_TITLE = "#d95700"
TEXT = "#111111"

# ---------- Helpers ----------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")  # keep empty strings
    if "guests" in df.columns:
        df["guests"] = df["guests"].apply(lambda x: x if not (isinstance(x, str) and x.startswith("[") and x.endswith("]")) else re.sub(r"[\[\]']","",x).strip())
    return df

def _choose_first_existing_col(df: pd.DataFrame, candidates):
    """Return the first candidate column name that exists in df, or None."""
    cols_lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lc:
            return cols_lc[cand]
    # fallback to substring match (helps for slight naming variants)
    for cand in candidates:
        for actual in df.columns:
            if cand in actual.lower():
                return actual
    return None

def extract_episode_number(epid: str) -> Optional[int]:
    m = re.search(r"ep[-_ ]?(\d+)", str(epid))
    if m:
        return int(m.group(1))
    return None

def parse_date_or_episode_sort_key(row):
    if "GuestAppearanceDate" in row and row["GuestAppearanceDate"]:
        try:
            return pd.to_datetime(row["GuestAppearanceDate"], errors="coerce")
        except Exception:
            pass
    epn = extract_episode_number(row.get("Episode ID", ""))
    if epn is not None:
        return epn
    return None

# ---------- Load dataframe and detect expanded/highlight columns ----------
st.set_page_config(page_title="Off Menu — Mentions (MVP)", layout="wide")

if not Path(CSV_PATH).exists():
    st.error(f"CSV not found: {CSV_PATH}. Export your merged df to this path.")
    st.stop()

df = load_df(CSV_PATH)

# detect which columns to use for full / highlighted mention text
# candidate names (ordered preference)
HIGHLIGHT_COL_CANDIDATES = [
    "mention text highlighted",         # preferred (html with <mark>)
    "mention_text_highlighted",
    "mention_text_highlighted_html",
    "mention text highlighted html",
    "mention_text_highlighted_html"
]
FULL_COL_CANDIDATES = [
    "mention text full",
    "mention_text_full",
    "mentiontextfull",
    "mention full",
    "mention_full"
]
SHORT_COL_CANDIDATES = [
    "mention text",
    "mention_text",
    "mention"
]

HIGHLIGHT_COL = _choose_first_existing_col(df, HIGHLIGHT_COL_CANDIDATES)
FULL_COL = _choose_first_existing_col(df, FULL_COL_CANDIDATES)
SHORT_COL = _choose_first_existing_col(df, SHORT_COL_CANDIDATES)

def _get_preferred_mention_for_row(row):
    """
    Return a tuple (html_for_display, plain_text_for_search) for a mention row.
    - html_for_display: safe HTML to render (might contain <mark>), or escaped plain text if no highlight HTML.
    - plain_text_for_search: plain text used in filters/search (no HTML).
    """
    # Try highlighted HTML first (we assume it is already safe HTML with <mark>)
    if HIGHLIGHT_COL and row.get(HIGHLIGHT_COL):
        return row.get(HIGHLIGHT_COL), row.get(FULL_COL) or row.get(SHORT_COL) or ""
    # Then try full plain text (escape when rendering)
    if FULL_COL and row.get(FULL_COL):
        return html.escape(row.get(FULL_COL)), row.get(FULL_COL)
    # Fallback to short mention text
    if SHORT_COL and row.get(SHORT_COL):
        return html.escape(row.get(SHORT_COL)), row.get(SHORT_COL)
    return "", ""

# ---------- Aggregation (uses preferred mention columns) ----------
def make_restaurant_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_sort_key"] = df.apply(parse_date_or_episode_sort_key, axis=1)
    if "Timestamp" in df.columns:
        df["Timestamp_display"] = df["Timestamp"].replace("", "").astype(str)
    groups = []
    grouped = df.groupby(["region_header", "subtitle", "restaurant_key"], dropna=False)
    for (region, subtitle, rkey), g in grouped:
        mentions_count = len(g)
        try:
            latest_row = g.copy().sort_values("_sort_key", ascending=False, na_position="last").iloc[0]
        except Exception:
            latest_row = g.iloc[0]

        # Prefer highlighted HTML, else full text, else short text
        html_display, plain_text = _get_preferred_mention_for_row(latest_row)

        # Build list of all mentions with the preferred fields (store both html and plain for each)
        all_ments = []
        for _, mr in g.iterrows():
            m_html, m_plain = _get_preferred_mention_for_row(mr)
            all_ments.append({
                "EpisodeID": mr.get("Episode ID",""),
                "MentionTextPlain": m_plain,
                "MentionTextHTML": m_html,
                "Timestamp": mr.get("Timestamp",""),
                "Guest": mr.get("guests","")
            })

        groups.append({
            "region_header": region,
            "subtitle": subtitle or "",
            "restaurant": latest_row.get("restaurant", ""),
            "restaurant_key": rkey,
            "guests": latest_row.get("guests", ""),
            "mentions_count": mentions_count,
            "latest_mention_text_html": html_display,   # may contain <mark> or be escaped plain text
            "latest_mention_text_plain": plain_text,    # plain text for search/filtering
            "latest_timestamp": latest_row.get("Timestamp", ""),
            "_latest_sort_key": latest_row.get("_sort_key", None),
            "all_mentions": all_ments
        })
    return pd.DataFrame(groups)

# ---------- Streamlit UI & CSS (with mark styling) ----------
st.markdown(f"""
    <style>
    :root {{
      --om-bg: {OFFMENU_BG};
      --om-orange: {OFFMENU_TITLE};
      --om-text: {TEXT};
    }}
    .stApp {{ background: var(--om-bg); color: var(--om-text); }}
    .card {{
        background: white;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }}
    .restaurant-name {{ font-size:18px; font-weight:700; color: var(--om-text); }}
    .meta {{ font-size:13px; color:#555; margin-bottom:8px; }}
    .snippet {{ font-size:14px; color:#222; margin-bottom:8px; white-space:pre-wrap; }}
    .badge {{ background:var(--om-orange); color:white; padding:2px 8px; border-radius:10px; font-weight:700; font-size:12px; }}
    .region-title {{ font-size:22px; font-weight:800; margin-top:10px; margin-bottom:6px; color:var(--om-orange); }}
    .subtitle-title {{ font-size:16px; font-weight:700; margin-top:12px; margin-bottom:6px; }}
    mark {{
      background: #fff1c9;  /* gentle pale yellow */
      padding: 0 2px;
      border-radius: 2px;
    }}
    .main-col {{ width:100%; }}
    </style>
""", unsafe_allow_html=True)

st.title("Off Menu — Top mentions (MVP)")

# Controls
with st.sidebar:
    q = st.text_input("Search (restaurant / guest / snippet)", value="")
    st.header("Controls")
    include_nones = st.checkbox("Include restaurants with no mentions (show 'None')", value=False)
    show_all_mentions = st.checkbox("Show all mentions for each restaurant (expander)", value=False)
    include_final_10pct = st.checkbox(
        "Include matches from the final 10% of episode text", 
        value=False # Default is False, so they are excluded
    )
    all_regions = sorted(df["region_header"].unique())
    selected_regions = st.multiselect("Show regions", options=all_regions, default=all_regions)

# Filter None mentions if user wants
if not include_nones:
    # Use plain text column for filtering, but if it's empty rely on short mention column
    # We'll filter rows that have neither a short nor a full mention
    df = df[ (df.get(SHORT_COL, "") != "") | (df.get(FULL_COL, "") != "") | (df.get(HIGHLIGHT_COL, "") != "") ]

# Filter final 10% matches (likely from roundup and not good matches)
if not include_final_10pct:
    # Assuming 'match_in_final_10pct' is a boolean column (True/False) or a string column 
    # where "1" or "True" means it's in the final 10%. We'll filter to keep only non-matches.
    df = df[ (df.get("match_in_final_10pct", "0").astype(str).str.lower() != "true") & 
             (df.get("match_in_final_10pct", "0").astype(str).str.lower() != "1") ]

# Basic search (search in restaurant, guests, and preferred mention plain text)
if q:
    qlow = q.lower()
    df = df[df.apply(lambda r: qlow in str(r.get("restaurant","")).lower() or
                              qlow in str(r.get("guests","")).lower() or
                              qlow in str(r.get(SHORT_COL,"")).lower() or
                              qlow in str(r.get(FULL_COL,"")).lower(), axis=1)]

# Filter regions
df = df[df["region_header"].isin(selected_regions)]

# Build aggregated restaurant dataframe
agg_df = make_restaurant_aggregates(df)

# ---------- Ordering / London-first ----------
def ordered_subtitles_for_region(region_name, df_region):
    subs = sorted(df_region["subtitle"].fillna("").unique(), key=lambda s: s.lower())
    london_items = [s for s in subs if s.lower().startswith("london")]
    others = [s for s in subs if not s.lower().startswith("london")]
    ordered = sorted(london_items, key=lambda s: s.lower()) + sorted(others, key=lambda s: s.lower())
    return ordered

region_groups = {}
for region in sorted(agg_df["region_header"].unique(), key=lambda r: r.lower()):
    region_df = agg_df[agg_df["region_header"] == region].copy()
    region_groups[region] = ordered_subtitles_for_region(region, region_df)

regions = list(region_groups.keys())
left_region = regions[0] if regions else None
right_region = regions[1] if len(regions) > 1 else None

colL, colR = st.columns([1,1])

def render_region_in_column(region, col):
    if not region:
        return
    with col:
        st.markdown(f"<div class='region-title'>{html.escape(region)}</div>", unsafe_allow_html=True)
        subs = region_groups.get(region, [])
        for sub in subs:
            st.markdown(f"<div class='subtitle-title'>{html.escape(sub)}</div>", unsafe_allow_html=True)
            sub_df = agg_df[(agg_df["region_header"]==region) & (agg_df["subtitle"]==sub)].copy()
            sub_df = sub_df.sort_values(by="_latest_sort_key", ascending=True, na_position="last")

            for i, r in sub_df.iterrows():
                # r['latest_mention_text_html'] may already be HTML with <mark>, or an escaped string
                html_snippet = r.get("latest_mention_text_html", "") or ""
                # If the html_snippet does not contain any <mark> but is escaped plain text, display as-is (safe)
                st.markdown(f"""
                    <div class="card main-col">
                      <div class="restaurant-name">{html.escape(r.get('restaurant','') or '')} <span style="font-weight:600; font-size:12px; color:#666;">{html.escape(str(r.get('guests','')))}</span></div>
                      <div class="meta"><span class="badge">{r['mentions_count']}</span> &nbsp; {html.escape(str(r.get('latest_timestamp','')))}</div>
                      <div class="snippet">{html_snippet}</div>
                    </div>
                """, unsafe_allow_html=True)

                if show_all_mentions:
                    with st.expander("All mentions"):
                        for m in r.get("all_mentions", []):
                            # for each mention, prefer MentionTextHTML if present, else plain
                            m_html = m.get("MentionTextHTML", "")
                            if m_html:
                                st.markdown(m_html, unsafe_allow_html=True)
                            else:
                                st.markdown(f"- **{html.escape(str(m.get('EpisodeID','')))}** — {html.escape(str(m.get('MentionTextPlain','')))} ({html.escape(str(m.get('Timestamp','')))}).")

# Render
render_region_in_column(left_region, colL)
if right_region:
    render_region_in_column(right_region, colR)
