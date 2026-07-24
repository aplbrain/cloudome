# Running on AWS

This implementation uses the AWS tools SQS, Lambda (deployed with Zappa), and DynamoDB. Jobs are queued to SQS and executed by many parallel Lambdas. Results are stored and queryable in DynamoDB.

## Prerequisites

An AWS account and the AWS CLI configured on your machine.

## Install and Configure

First you will need to edit the `zappa_settings.json`. You will need to create an SQS queue whose ARN can be filled in on line 25.

Next you will need to deploy Cloudome onto Lambda using Zappa. These commands were run on Ubuntu. MacOS (ARM) won't work.

```bash
uv run zappa deploy
uv run zappa schedule
# then for incremental updates,
uv run zappa update
```

## Provision Resources

To initialize required resources such as DynamoDB tables, run:

```bash
uv run python manage.py initialize
```

## generate a contactome

### populate the task queue

```bash
uv run python manage.py contactome generate # optionally test with --enqueue-limit 1
```

### wait...

```bash
# this will someday go to zero:
AWS_REGION=us-east-1 aws sqs get-queue-attributes --queue-url "https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs" --attribute-names ApproximateNumberOfMessagesNotVisible
```

### collect results

```bash
uv run python manage.py export example_graph_id contactome_40k.csv
```

### convert to a weighted contactome edgelist

To convert the raw contactome data into a weighted edgelist, use the `simplify` command:

```bash
uv run python manage.py contactome simplify --raw-file contactome_40k.csv --output-file pre_post_weights.csv
```

This will leave you with `pre_post_weights.csv`, which contains aggregated weights for each (pre, post) pair.

## compute per-segment volume

End-to-end flow for computing voxel counts per segmentation ID across the volume.

Global flags: you can pass a global `--mip` (single int or comma-separated) and `--sqs-url` to all commands below.

### populate the task queue

Enqueue cuboid-wise volume tasks over your segmentation channel. Adjust block sizes and Z range as needed.

```bash
uv run python manage.py volume generate \
	--graph-id volume-40k \
	--segmentation-channel s3://path/to/segmentation/ \
	--block-size-x 64 --block-size-y 64 --block-size-z 32 \
	--z-start 0 --z-end 1000 \
	--enqueue-limit 10  # optional for a quick smoke test
```

Each task processes a cuboid and the worker emits DynamoDB items like:

```
graph_id=volume-40k
synapse_id=vol_x{xs}_y{ys}_z{zs}_seg{SEGID}_v{VOXEL_COUNT}
```

### wait...

Monitor queue depth while workers process jobs:

```bash
AWS_REGION=us-east-1 aws sqs get-queue-attributes \
	--queue-url "https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs" \
	--attribute-names ApproximateNumberOfMessagesNotVisible
```

### collect results

Export raw results for a given `graph_id` to CSV:

```bash
uv run python manage.py export volume-40k volume_raw.csv
```

### simplify to per-seg totals

Aggregate voxel counts per segmentation ID from the exported CSV:

```bash
uv run python manage.py volume simplify \
	--raw-file volume_raw.csv \
	--output-file seg_voxel_counts.csv
```

This produces `seg_voxel_counts.csv` with columns:

```
seg_id,voxel_count
```

## generate a connectome

```bash
uv run python manage.py synapses generate --synapse-channel s3://path/to/synapses/ --output-file synapse-centroids-40k.csv --mask post

uv run python manage.py synapses enqueue --graph-id connectome-40k --centroids-file synapse-centroids-40k.csv --synapse-channel s3://path/to/synapses/ --segmentation-channel s3://path/to/segmentation/ --enqueue-limit 10

uv run python manage.py export connectome-40k synapses_40k.csv

uv run python manage.py synapses simplify --raw-file synapses_40k.csv --output-file synapse_weights.csv
```
