# cloudome

## install and configure

```bash
zappa deploy
zappa schedule
# then for incremental updates,
zappa update
```

## generate a contactome

### populate the task queue

```bash
uv run python3 ./enqueue_centroids.py contactome generate # optionally test with --enqueue-limit 1
```

### wait...

```bash
# this will someday go to zero:
AWS_REGION=us-east-1 aws sqs get-queue-attributes --queue-url "https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs" --attribute-names ApproximateNumberOfMessagesNotVisible
```


### collect results

```bash
uv run python3 ./enqueue_centroids.py export example_graph_id contactome_40k.csv
```

### convert to a weighted contactome edgelist

```bash
sh process_contactome.sh
```

This will leave you with `pre_post_weights.csv`.

## debugging

- needed to delete crc32c import in exceptions.py of cloudvolume