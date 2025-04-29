"""
Script to collect Reddit data from a given subreddit and year. For other parameters, see below.

Example usage:

> python3 collect_reddit.py --
> Saved to data/reddit/out.parquet
"""

import argparse
from utils import RedditGranularity, RedditSortBy, getRedditPostsPraw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for a single run of Reddit data collection.")
    parser.add_argument("-s", "--subreddit", type=str, help="Used to specify the name of the subreddit to scrape (no r/!)", default="depression")
    parser.add_argument("-N", "--topN", type=int, help="Used to define the max number of posts we are interested in considering", default=1000)
    parser.add_argument("-c", "--comments", type=int, help="Defines the maximum number of comments per post we are interested in retrieving.", default=50)
    parser.add_argument("-y", "--year", type=int, help="Defines the year from which we want to narrow our search.", default=None)
    parser.add_argument("-o", "--sort", type=str, choices=["new", "old", "top", "controversial"], help="Determines the sorting policy for the comments.", default="controversial")
    parser.add_argument("-g", "--granularity", type=str, choices=["topmonth", "topyear", "topalltime", "hot", "searchOzempic"], help="Determines the granularity of the search.", default="searchOzempic")
    args = parser.parse_args()

    getRedditPostsPraw(
        subreddit=args.subreddit,
        topN=args.topN,
        comments_per_post=args.comments,
        sortBy=RedditSortBy(args.sort),
        granularity=RedditGranularity(args.granularity),
    )
