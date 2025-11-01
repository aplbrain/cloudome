# Local Use

## Contactomes

To generate contactomes and connectomes locally, you can use the following commands. This will not deploy any cloud resources (no SQS, no Lambdas).

### Provision the queue:

```bash
uv run local_manage.py contactome generate --graph-id test0 --segmentation-channel s3://cvdb-bossdb-boss/martinez2025/zebrafish/shuffle_1_checkpoint_5000
```

## Connectomes

### Generate synapse centroids file:

```bash
uv run local_manage.py synapses generate --synapse-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses --output-file synapse-centroids-test0.csv
```

### Provision the queue:

```bash
uv run local_manage.py --sqlite-db-path synapse_edges_test0.db synapses enqueue --graph-id connectome-test0 --centroids-file synapse-centroids-test0.csv --synapse-channel s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses --segmentation-channel s3://cvdb-bossdb-boss/martinez2025/zebrafish/shuffle_1_checkpoint_5000
```

### Sharding SQLite DB

On highly parallel systems such as shared-filesystem clusters, you will likely want to shard the sqlite DB. Pass `--shard-sqlite` to any command that takes a database path (e.g. `--sqlite-db-path synapse_edges_test0.db --shard-sqlite`). This will create a unique sqlite DB per worker process.

## Running workers

```bash
uv run local_manage.py worker
```

You can also limit the number of tasks processed by each worker for testing purposes:

```bash
uv run local_manage.py worker --dequeue-limit 10
```
