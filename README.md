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
uv run python3 manage.py contactome generate # optionally test with --enqueue-limit 1
```

### wait...

```bash
# this will someday go to zero:
AWS_REGION=us-east-1 aws sqs get-queue-attributes --queue-url "https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs" --attribute-names ApproximateNumberOfMessagesNotVisible
```

### collect results

```bash
uv run python3 manage.py export example_graph_id contactome_40k.csv
```

### convert to a weighted contactome edgelist

```bash
sh process_contactome.sh
```

This will leave you with `pre_post_weights.csv`.

### can also do simplify.ipynb :)

## debugging

-   needed to delete crc32c import in exceptions.py of cloudvolume

## generate a connectome

```bash
uv run manage.py synapses generate --synapse-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses/ --output-file synapse-centroids-40k.csv

uv run manage.py synapses enqueue --graph-id connectome-40k --centroids-file synapse-centroids-40k.csv --synapse-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses/ --segmentation-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/agglomeration_checkpoint_40000/ --enqueue-limit 10
```

# options for upscale

1. run the generate step on big machine
2. generate at one mip, then do some math in the CSV, run the enqueue at a different MIP
