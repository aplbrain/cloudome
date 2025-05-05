# cloudome

## install and configure

```bash
zappa deploy
zappa schedule
# then for incremental updates,
zappa update
```

## provisioning resources

To initialize required resources such as DynamoDB tables, run:

```bash
uv run python3 manage.py initialize
```

Note: SQS queues are not yet provisioned automatically and must be created manually.

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

To convert the raw contactome data into a weighted edgelist, use the `simplify` command:

```bash
uv run python3 manage.py contactome simplify --raw-file contactome_40k.csv --output-file pre_post_weights.csv
```

This will leave you with `pre_post_weights.csv`, which contains aggregated weights for each (pre, post) pair.

## generate a connectome

```bash
uv run manage.py synapses generate --synapse-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses/ --output-file synapse-centroids-40k.csv

uv run manage.py synapses enqueue --graph-id connectome-40k --centroids-file synapse-centroids-40k.csv --synapse-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses/ --segmentation-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/agglomeration_checkpoint_40000/ --enqueue-limit 10

uv run manage.py synapses export connectome-40k synapses_40k.csv

uv run manage.py synapses simplify --raw-file synapses_40k.csv --output-file synapse_weights.csv
```
