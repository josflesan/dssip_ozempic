"""
Utility scripts for data collectiona and wrangling
"""

import logging
import os
import re
from datetime import datetime
from datetime import timezone
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import praw
import prawcore
import pytz
import time
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

########################################################################
# CONSTANTS
########################################################################

CHARACTER_LIMIT = 2000  # N characters top per comment will be considered
NUM_WORKERS = 8

########################################################################
# DATA AND SECRETS
########################################################################

# Load environment variables from secrets.env
load_dotenv("secrets.env")
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
app_name = os.getenv("APP_NAME")
secret_id = os.getenv("SECRET_ID")
client_id = os.getenv("CLIENT_ID")

# Time manipulation
local = pytz.timezone("Europe/Zurich")

class RedditGranularity(Enum):
    TOP_MONTH = "topmonth"
    TOP_YEAR = "topyear"
    TOP_ALLTIME = "topalltime"
    HOT = "hot"
    SEARCH = "searchOzempic"

class RedditSortBy(Enum):
    NEW = "new"
    OLD = "old"
    TOP = "top"
    CONTROVERSIAL = "controversial"


########################################################################
# MISC FUNCTIONS
########################################################################


def normalize_markdown(md_string: str):
    # Normalize headings (e.g., remove extra spaces after '#' and standardize levels)
    md_string = re.sub(r"^\s*(#+)\s*", r"\1 ", md_string, flags=re.MULTILINE)

    # Normalize lists (remove extra spaces)
    md_string = re.sub(
        r"^\s*[-\*\+]\s+", "- ", md_string, flags=re.MULTILINE
    )  # Bullet lists
    md_string = re.sub(
        r"^\s*\d+\.\s+", "1. ", md_string, flags=re.MULTILINE
    )  # Numbered lists

    # Normalize code blocks and inline code (add spaces around inline code)
    md_string = re.sub(
        r"`([^`]+)`", r"`\1`", md_string
    )  # Inline code: normalize spaces

    # Remove extra blank lines (to ensure consistent line breaks)
    md_string = re.sub(r"\n\s*\n", "\n\n", md_string)

    # Optional: Normalize links by ensuring they are in the form [text](url)
    md_string = re.sub(
        r"!\[([^\]]+)\]\(([^\)]+)\)", r"![\1](\2)", md_string
    )  # Image links

    return md_string.strip()

def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    flattened_rows = []

    for idx, row in df.iterrows():
        flattened_rows.append({
            "type": "post",
            "post_index": idx,
            "content": f"{row['title']}. {row['content']}",
            "upvotes": row['upvotes'],
            "timestamp": row['timestamp']
        })

        # Add each comment as its own row
        for comment in row.get("comments", []):
            flattened_rows.append({
                "type": "comment",
                "post_index": idx,
                "content": comment['content'],
                "upvotes": comment["upvotes"],
                "timestamp": comment['timestamp']
            })

    return pd.DataFrame(flattened_rows)

def convertLabelToSentiment(label: str):
    match label:
        case "LABEL_0":
            return "Negative"
        case "LABEL_1":
            return "Neutral"
        case "LABEL_2":
            return "Positive"


########################################################################
# REDDIT FUNCTIONS
########################################################################

reddit = praw.Reddit(client_id=client_id, client_secret=secret_id, user_agent=app_name)

def getRedditPostsPraw(
    subreddit: str,
    topN: int,
    comments_per_post: int,
    sortBy: RedditSortBy,
    granularity: RedditGranularity,
    out_path: str = "data/reddit",
):
    """
    Returns a list of the comments from the top N posts within the given granularity
    as a CSV, with one row per comment and some additional metadata (TODO: what metadata?)

    Args:
        subreddit (str): the name of the subreddit we are accessing
        topN (int): the top number of posts that we are interested in
        comments_per_post (int): the number of comments per post we are scraping
        granularity (RedditGranularity): the granularity of the search
        out_path (str): the output folder to which we write our parquet file
    """

    # Load posts from the subreddit
    sub = reddit.subreddit(subreddit)
    match granularity:
        case RedditGranularity.TOP_MONTH:
            posts = sub.top(time_filter="month", limit=topN)
        case RedditGranularity.TOP_YEAR:
            posts = sub.top(time_filter="year", limit=topN)
        case RedditGranularity.TOP_ALLTIME:
            posts = sub.top(time_filter="all", limit=topN)
        case RedditGranularity.HOT:
            posts = sub.hot(limit=topN)
        case RedditGranularity.SEARCH:
            posts = sub.search("ozempic", sort=sortBy, time_filter="all", limit=topN)

    posts = list(posts)
    logger.info(f"FOUND {len(posts)} POSTS")
    logger.info("COLLECTING POSTS...")

    scraped_data = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(_process_post, post, comments_per_post, sortBy) for idx, post in enumerate(posts)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing posts"):
            result = future.result()
            scraped_data.append(result)

    # Convert data into Pandas DataFrame and export
    filename = f"{subreddit}_top{topN if topN < len(posts) else 'all'}_{granularity.value}"
    scraped_df = pd.DataFrame(scraped_data)
    scraped_df = scraped_df.fillna(
        {
            "distinguished": 0,
            "upvotes": 0,
            "upvote_ratio": 0,
            "comment_replies": 0,
            "comment_upvotes": 0,
        }
    )
    scraped_df.to_parquet(
        f"{out_path}/{filename}.parquet", engine="pyarrow", index=False
    )

    logger.info(f"Data saved to {out_path}/{filename}")

def _process_post(submission, comments_per_post, sortBy):    
    post_data = {
        "title": submission.title,
        "content": submission.selftext,
        "distinguished": submission.distinguished
        if submission.distinguished
        else "False",
        "upvotes": submission.score,
        "upvote_ratio": submission.upvote_ratio,
        "timestamp": submission.created_utc,
        "comments": [],
    }

    try:
        submission.comment_sort = sortBy  # Sort the comments by newest
        submission.comments.replace_more(limit=None)
        comments_list = list(submission.comments.list())
        count = 0

        for comment in comments_list:
            # Skip the comment if the author deleted
            if comment.body[:100] in ["[deleted]", "[removed]"]:
                continue

            # Format datetime
            utc_datetime = datetime.fromtimestamp(comment.created_utc).isoformat()

            post_data["comments"].append(
                {
                    "timestamp": comment.created_utc,
                    "datetime": utc_datetime,
                    "content": normalize_markdown(comment.body[:CHARACTER_LIMIT]),
                    "distinguished": comment.distinguished
                    if comment.distinguished
                    else "False",
                    "replies": len(comment.replies.list()) if comment.replies else 0,
                    "upvotes": comment.score,
                }
            )

            count += 1
            if count == (comments_per_post - 1):
                break

    except Exception as e:
        logger.warning(f"Failed to process submission {submission.id}: {e}")

    return post_data
