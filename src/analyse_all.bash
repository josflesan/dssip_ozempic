#!/bin/bash

# List of files to analyze
files=("depression_topall_searchOzempic" "celebritygossip_topall_topalltime" "EatingDisorders_topall_searchOzempic" "nutrition_topall_searchOzempic")

# Loop through each file and run the Python script
for file in "${files[@]}"
do
    echo "NLP Analysis on data from $file..."
    python3 src/nlp_analysis.py \
        --file "data/reddit/${file}.parquet"
done
