#!/bin/bash

# List of subreddits to collect data from
subreddits=("loseit" "diabetes" "Instagramreality" "fatlogic" "nutrition" "EatingDisorders" "depression")

# Loop through each subreddit and run the Python script
for subreddit in "${subreddits[@]}"
do
    echo "Collecting data from /r/$subreddit..."
    python3 src/collect_reddit.py \
        --subreddit "$subreddit" \
        --topN 1000 \
        --comments 10 \
        --granularity "topalltime" \
        --sort "new"
done
