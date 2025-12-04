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
import logging
from pathlib import Path
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Optional
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import shutil

from .config import episodes_list_url, restaurants_url, transcript_base_url

from .utils import save_text_to_file, extract_html, try_read_parquet

logger = logging.getLogger("scraper")

# =========================================================================
# 2. Configuration (paths, constants, etc.)
# =========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
test_temp_dir = os.path.join(project_root, "data/test_temp")
raw_data_path = os.path.join(project_root, "data/raw")
processed_data_path = os.path.join(project_root, "data/processed")

# =========================================================================
# 3. Helper Functions
# =========================================================================


# Function to set up logger for new web scraper
def configure_logger(log_file: Optional[str] = None, level: int = logging.DEBUG):
    """
    Configure a compact logger for the scraper.
    - Console handler always enabled.
    - Optional file handler if log_file provided.
    - Default level: DEBUG for maximum visibility while testing.
    """
    logger = logging.getLogger("scraper")
    logger.setLevel(level)

    # Avoid adding handlers multiple times when running multiple times in a notebook
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler (clear, one-line format)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)

    # Optional file handler (rotating not necessary here — keep simple)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)

    return logger


# small sanitize helper (same as before)
def _sanitize_key(key: str) -> str:
    if not isinstance(key, str):
        key = str(key)
    s = key.strip().lower()
    s = re.sub(r"[^a-z0-9\-_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-_")


# ---- Helper: random-ish UA list (small) ----
_SIMPLE_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]


def _choose_headers():
    return {"User-Agent": random.choice(_SIMPLE_USER_AGENTS)}


# ----- Helper to access retry limits from the server (for use in scraper)
def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """
    Parse Retry-After header. It can be:
      - an integer number of seconds, e.g. "120"
      - a HTTP-date string, e.g. "Wed, 21 Oct 2015 07:28:00 GMT"
    Return number of seconds to wait (float), or None if not parseable.
    """
    if not header_value:
        return None
    header_value = header_value.strip()
    # try integer seconds
    if header_value.isdigit():
        try:
            return float(header_value)
        except Exception:
            return None
    # try HTTP-date
    try:
        dt = parsedate_to_datetime(header_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = (dt - now).total_seconds()
        return max(0.0, float(delta))
    except Exception:
        return None


# Helper function to assimilate old legacy transcripts (find them, update status.json, rename them to new system)


def assimilate_existing_transcripts(
    out_dir: Path,
    url_map: Dict[str, str],
    status: Dict[str, Dict],
    legacy_dir: Optional[Path] = None,
    rename_to_slug: bool = True,
    overwrite: bool = False,
) -> Dict[str, Dict]:
    """
    Find legacy files named like 'ep_1.html' or 'ep-1.html' in out_dir (and optionally legacy_dir),
    map them to url_map keys (slugs that contain 'ep-<num>' or '<num>'), update the status dict and
    optionally rename/move them to the slug-based filename.

    Args:
        out_dir: Path where new slug files should live.
        url_map: mapping slug -> url (used to find matching slug for an ep number).
        status: status dict to update in-place (returned for convenience).
        legacy_dir: optional extra directory to check for files (if your old files live elsewhere).
        rename_to_slug: if True, move/rename legacy file to new slug filename (safe move).
        overwrite: if True, allow overwriting existing slug files (be careful).

    Returns:
        Updated status dict (mutated in-place).
    """
    out_dir = Path(out_dir)
    candidates = []

    _LEGACY_EP_RE = re.compile(r"ep[_\-]?(\d+)\.html$", flags=re.IGNORECASE)
    # collect files to examine from out_dir
    for p in out_dir.glob("*.html"):
        candidates.append(p)

    # also check legacy_dir if provided
    if legacy_dir:
        legacy_dir = Path(legacy_dir)
        if legacy_dir.exists():
            for p in legacy_dir.glob("*.html"):
                # avoid double-adding files that are already in out_dir (same path)
                if p.resolve() not in [c.resolve() for c in candidates]:
                    candidates.append(p)

    # build reverse map: number_str -> list of slugs that contain that number token
    # e.g. '1' -> ['ep-1-john-doe', 'ep-1-other']
    num_to_slugs = {}
    for slug in url_map.keys():
        # find first number token like ep-<num> or ep<num>
        m = re.search(r"ep[-_]?(?P<num>\d+)", slug, flags=re.IGNORECASE)
        if m:
            num = m.group("num")
            num_to_slugs.setdefault(num, []).append(slug)
        else:
            # also consider bare numbers anywhere (e.g. 'episode-23-guest')
            m2 = re.search(r"(?<!\d)(\d+)(?!\d)", slug)
            if m2:
                num = m2.group(1)
                num_to_slugs.setdefault(num, []).append(slug)

    summary = {"found": 0, "mapped": 0, "renamed": 0, "skipped": 0}

    for p in candidates:
        name = p.name
        m = _LEGACY_EP_RE.match(name)
        if not m:
            # not a legacy ep_N file; ignore
            continue
        summary["found"] += 1
        epnum = m.group(1)

        # find candidate slugs for this ep number
        candidates_for_num = num_to_slugs.get(epnum, [])

        if not candidates_for_num:
            # no matching slug for the number — skip for now
            logger.debug(
                "Found legacy file %s but no slug contains ep-%s; skipping", name, epnum
            )
            summary["skipped"] += 1
            continue

        # If multiple slugs match one number, prefer exact 'ep-<num>' prefix match
        chosen_slug = None
        for s in candidates_for_num:
            if re.match(rf"^ep[-_]?{epnum}(\b|-|$)", s, flags=re.IGNORECASE):
                chosen_slug = s
                break
        if chosen_slug is None:
            chosen_slug = candidates_for_num[0]

        # Build destination path for slug file
        safe_slug = _sanitize_key(chosen_slug)
        dest = out_dir / f"{safe_slug}.html"

        # If dest already exists and is same file, just update status
        try:
            if dest.exists() and dest.resolve() == p.resolve():
                logger.debug("Legacy file %s already at desired location %s", p, dest)
            elif dest.exists() and not overwrite:
                # dest already exists (someone downloaded or moved it earlier) -> skip rename but update status to point to dest
                logger.info(
                    "Destination %s exists; skipping move of %s (overwrite=False)",
                    dest,
                    p,
                )
                summary["skipped"] += 1
            else:
                if rename_to_slug:
                    # move (or copy+unlink) legacy file to dest in a safe manner
                    logger.info("Renaming/moving legacy file %s -> %s", p, dest)
                    # ensure parent exists
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    # use shutil.move to preserve contents; if same filesystem this is cheap
                    shutil.move(str(p), str(dest))
                    summary["renamed"] += 1
                else:
                    # don't move but use p as saved_path
                    dest = p
        except Exception as e:
            logger.exception("Failed to move/inspect legacy file %s: %s", p, e)
            summary["skipped"] += 1
            continue

        # Update status entry for chosen slug
        meta = status.setdefault(
            chosen_slug,
            {
                "url": url_map.get(chosen_slug),
                "attempts": 0,
                "status": "pending",
                "saved_path": None,
                "last_error": None,
            },
        )
        meta.update(
            {
                "attempts": max(meta.get("attempts", 0), 1),
                "status": "success",
                "saved_path": str(dest),
                "last_error": None,
            }
        )
        logger.info(
            "Associated legacy file %s -> slug=%s saved_path=%s",
            name,
            chosen_slug,
            dest,
        )
        summary["mapped"] += 1

    logger.info("Assimilation summary: %s", summary)
    return status


# Function to download (scrape) the transcripts, legacy compatible version
# checks a directory (see legacy_dir = ... in the function) for old style transcripts
# and updates the status.json accordingly, and renames the files, then skips them as they're already downloaded
# replaces download transcripts which replaced _save_transcripts_html
def download_transcripts_legacy(
    url_map: Dict[str, str],
    out_dir: str,
    status_path: str,
    max_attempts_per_url: int = 5,
    backoff_base: float = 1.0,
    max_workers: int = 3,
    session: Optional[requests.Session] = None,
    timeout: float = 12.0,
    legacy_dir=None,
) -> Dict[str, Dict]:
    """
    Download URLs to out_dir using url_map (keys are slugs used as filenames).
    Added logging provides visibility into what the function does on each run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(status_path)

    logger.info(
        "Starting download_transcripts: %d urls, out_dir=%s, status_path=%s",
        len(url_map),
        out_dir,
        status_path,
    )

    # Load existing status if present (allows resume)
    if status_path.exists():
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            logger.debug("Loaded existing status.json with %d entries", len(status))
        except Exception as e:
            logger.warning(
                "Failed to load status.json (%s). Starting with empty status.", e
            )
            status = {}
    else:
        logger.debug("No status.json file found at %s. Starting fresh.", status_path)
        status = {}

    # Initialize status entries for any missing keys (log each new init)
    for key, url in url_map.items():
        if key not in status:
            status[key] = {
                "url": url,
                "attempts": 0,
                "status": "pending",  # pending | success | failed
                "saved_path": None,
                "last_error": None,
            }
            logger.debug("Initialized status for key='%s' -> %s", key, url)

    # try assimilating legacy files in out_dir and a legacy folder (if you have one)
    if legacy_dir:
        status = assimilate_existing_transcripts(
            out_dir=out_dir,
            url_map=url_map,
            status=status,
            legacy_dir=legacy_dir,
            rename_to_slug=True,
            overwrite=False,
        )
        # persist immediately so the status file reflects these existing files
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        logger.debug("Persisted status.json after assimilating legacy transcripts.")

    # Use a single session for pooling
    session = session or requests.Session()

    def _attempt_download(key: str, meta: Dict) -> Dict:
        url = meta["url"]
        attempts = meta["attempts"]
        result = dict(meta)

        # If already succeeded, skip and log reason
        if meta.get("status") == "success":
            logger.debug(
                "Skipping key='%s' (already success, saved_path=%s)",
                key,
                meta.get("saved_path"),
            )
            return result

        # If max attempts reached, log and skip
        if attempts >= max_attempts_per_url:
            result["status"] = "failed"
            result["last_error"] = "max_attempts_reached"
            logger.info(
                "Key='%s' reached max attempts (%d). Marking failed.", key, attempts
            )
            return result

        # Log the attempt about to be made
        logger.debug("Attempting key='%s' (attempt %d) -> %s", key, attempts + 1, url)
        try:
            headers = _choose_headers()
            resp = session.get(url, headers=headers, timeout=timeout)

            # If success (200)
            if resp.status_code == 200:
                safe_key = _sanitize_key(key)
                filename = f"{safe_key}.html"
                saved_path = str(out_dir / filename)

                # If file already exists, log that we're overwriting (helps debug)
                if Path(saved_path).exists():
                    logger.debug(
                        "File %s already exists and will be overwritten by key='%s'",
                        saved_path,
                        key,
                    )

                with open(saved_path, "w", encoding="utf-8") as fh:
                    fh.write(resp.text)

                result.update(
                    {
                        "attempts": attempts + 1,
                        "status": "success",
                        "saved_path": saved_path,
                        "last_error": None,
                    }
                )
                logger.info("Saved %s -> %s (key=%s)", url, saved_path, key)
                return result

            # Retryable status codes
            if resp.status_code in (429, 500, 502, 503, 504):
                # Parse Retry-After header if present and include in result
                retry_after_raw = resp.headers.get("Retry-After")
                retry_after_seconds = _parse_retry_after(retry_after_raw)
                result.update(
                    {
                        "attempts": attempts + 1,
                        "status": "pending",
                        "last_error": f"status_{resp.status_code}",
                        "retry_after_seconds": retry_after_seconds,
                    }
                )
                logger.warning(
                    "Retryable HTTP %s for key='%s' url=%s (attempt %s)",
                    resp.status_code,
                    key,
                    url,
                    attempts + 1,
                )
                # Log headers optionally for 429 to see Retry-After
                if resp.status_code == 429:
                    logger.debug(
                        "429 response headers for key='%s': Retry-After=%s",
                        key,
                        retry_after_raw,
                    )
                    logger.debug(
                        "Parsed Retry-After seconds for key='%s': %s",
                        key,
                        retry_after_seconds,
                    )
                return result

            # Non-retryable
            result.update(
                {
                    "attempts": attempts + 1,
                    "status": "failed",
                    "last_error": f"status_{resp.status_code}",
                }
            )
            logger.error(
                "Non-retryable HTTP %s for key='%s' url=%s", resp.status_code, key, url
            )
            return result

        except requests.RequestException as e:
            # Network error: retryable
            result.update(
                {"attempts": attempts + 1, "status": "pending", "last_error": repr(e)}
            )
            logger.warning(
                "RequestException for key='%s' url=%s (attempt %s): %s",
                key,
                url,
                attempts + 1,
                e,
            )
            return result

    # Worker wrapper with backoff
    def _worker_task(key):
        meta = status[key]
        if (
            meta.get("status") == "success"
            or meta.get("attempts", 0) >= max_attempts_per_url
        ):
            return key, meta

        new_meta = _attempt_download(key, meta)

        if new_meta["status"] == "pending":
            # computed exponential backoff (what we would do)
            comp_sleep = backoff_base * (2 ** (new_meta["attempts"] - 1))
            jitter = random.uniform(0, 1.0)
            computed_sleep = comp_sleep + jitter

            # server-provided advice (if any)
            retry_after = new_meta.get("retry_after_seconds")
            if retry_after is not None:
                # use the server's suggestion if it's longer than our computed wait
                sleep_time = max(computed_sleep, float(retry_after))
            else:
                sleep_time = computed_sleep

            # cap to avoid runaway sleeps (adjust cap as desired)
            sleep_time = min(sleep_time, 600.0)

            logger.info(
                "Backing off %0.2fs for key='%s' (attempt %s) [computed=%0.2fs, server=%s]",
                sleep_time,
                key,
                new_meta["attempts"],
                computed_sleep,
                retry_after,
            )
            time.sleep(sleep_time)

        return key, new_meta

    # Main loop
    pending_keys = [
        k
        for k, v in status.items()
        if v["status"] != "success" and v["attempts"] < max_attempts_per_url
    ]
    round_idx = 0
    while pending_keys:
        round_idx += 1
        logger.info("Download round %d: %d pending", round_idx, len(pending_keys))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_worker_task, key): key for key in pending_keys}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    k, new_meta = fut.result()
                    status[k].update(new_meta)
                except Exception as e:
                    logger.exception("Unhandled exception for key %s: %s", key, e)
                    status[key]["attempts"] = status[key].get("attempts", 0) + 1
                    status[key]["last_error"] = repr(e)

        # persist status to disk after every round
        try:
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
            logger.debug("Persisted status.json (round %d).", round_idx)
        except Exception as e:
            logger.exception("Failed to write status file: %s", e)

        # Prepare next round
        pending_keys = [
            k
            for k, v in status.items()
            if v["status"] != "success" and v["attempts"] < max_attempts_per_url
        ]

        if pending_keys:
            logger.info("Sleeping 2s between rounds to be polite...")
            time.sleep(2)

    # final persist and summary
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    # Final summary counts
    succ = sum(1 for v in status.values() if v.get("status") == "success")
    failed = sum(1 for v in status.values() if v.get("status") == "failed")
    pending = sum(1 for v in status.values() if v.get("status") == "pending")
    logger.info(
        "Download finished. success=%d failed=%d pending=%d", succ, failed, pending
    )

    return status


# =========================================================================
# 4. Main Logic Functions
# =========================================================================


# Function to extract html (for episodes site and restaurants site)
def extract_and_save_html(site_url, output_html_filepath):
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


# Orchestration function for the scraper, to scrape transcripts
# Legacy compatible version (main edits in download_transcripts_legacy) to check for old style trasncripts and update filenames
# replaces orchestrate_scraper which replaced extract_and_save_transcripts_html
def orchestrate_scraper_legacy(
    df,  # DataFrame with 'slug' and optionally 'url'
    base_url,  # base URL for constructing URLs if df has no 'url' column
    out_dir,  # folder to save HTML transcripts
    max_attempts_per_url=5,
    backoff_base=1.0,
    max_workers=3,
    timeout=12.0,
    legacy_dir=None,
):
    """
    Orchestrates the scraping process:
      1. Prepares a slug → URL map
      2. Ensures output folder exists
      3. Calls download_transcripts() with sensible defaults
      4. Returns the status dict for all downloads

    Args:
        df: dataframe (not filepath) with slugs and urls in (also raw titles, guest names)
        base_url: The base url for the transcripts from podscripts.com
        out_dir: The folder to save the transcripts to
        max_attempts_per_url
        backoff_base
        max_workers
        timeout
    """
    # ---------------------
    # Setup logger for this run
    # ---------------------
    logger = configure_logger()
    logger.info("Starting scraper orchestration for %d episodes", len(df))

    # ---------------------
    # Prepare URL map
    # ---------------------
    if "url" in df.columns:
        url_map = {row["slug"]: row["url"] for _, row in df.iterrows()}
        logger.info("Using existing URLs from DataFrame")
    else:
        url_map = {
            row["slug"]: base_url.rstrip("/") + "/" + row["slug"].lstrip("/")
            for _, row in df.iterrows()
        }
        logger.info("Constructed URLs from base_url and slugs")

    # ---------------------
    # Ensure output folder exists
    # ---------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"

    # ---------------------
    # Call the scraper
    # ---------------------
    logger.info("Running download_transcripts with %d URLs", len(url_map))
    status = download_transcripts_legacy(
        url_map=url_map,
        out_dir=out_dir,
        status_path=status_path,
        max_attempts_per_url=max_attempts_per_url,
        backoff_base=backoff_base,
        max_workers=max_workers,
        timeout=timeout,
        legacy_dir=legacy_dir,
    )

    logger.info("Scraper orchestration finished")
    return status


# =========================================================================
# 5. Script exectuion
# This section contains script that runs only when this script is run directly when it is open (not when called by another script)
# This will contain a smaller model of the processes, so we can test before implementing in main
# =========================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Episode extraction
    # -------------------------------------------------------------------------
    save_text_to_file(extract_html(episodes_list_url), "episodes.html", test_temp_dir)

    # -------------------------------------------------------------------------
    # Restaurants extraction
    # -------------------------------------------------------------------------

    # Extract restaurants site html and store
    save_text_to_file(
        extract_html(restaurants_url), "restaurants_site.html", test_temp_dir
    )

    # -------------------------------------------------------------------------
    # Transcripts extraction
    # -------------------------------------------------------------------------
    # Note - we use dummy data here both to avoid having to depend on data processing, and bc purpose of testing

    # Opening dummy data (head of full dataframe) and testing extraction
    V2_tests_dir = os.path.join(test_temp_dir, "V2_tests")
    ep_meta_and_mentions_head_path = os.path.join(
        V2_tests_dir, "second_ten_test_episodes_full_metadata.parquet"
    )
    try:
        ep_meta_and_mentions_head_df = pd.read_parquet(ep_meta_and_mentions_head_path)
        # Testing on dummy data
        orchestrate_scraper_legacy(
            ep_meta_and_mentions_head_df, transcript_base_url, test_temp_dir
        )
    except FileNotFoundError:
        print(
            f"Error: The file was not found at {ep_meta_and_mentions_head_path}. Did it save correctly?"
        )
