#!/bin/bash

# This script processes the contactome_40k.csv file to generate a new list of lines in the format (pre, post, sum-of-weights).

# Use gawk for better compatibility
gawk -F, '
BEGIN {
    OFS = ",";
}
{
    # Extract pre, post, and weight (w) from the synapse_id field
    match($2, /pre([0-9]+)_post([0-9]+)_w([0-9]+)/, arr);
    pre = arr[1];
    post = arr[2];
    weight = arr[3];

    # Accumulate weights for each pre-post pair
    key = pre "," post;
    weights[key] += weight;
}
END {
    # Output the accumulated weights
    for (key in weights) {
        print key, weights[key];
    }
}' contactome_40k.csv > pre_post_weights.csv

# Notify the user of completion
echo "Processing complete. Results saved to pre_post_weights.csv."