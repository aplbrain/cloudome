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



## simplify a contactome SQLite database

If you already have a populated contactome SQLite database (for example the
worker output written by `local_manage.py`), you can aggregate it in-place or
write a new compact database that only retains `(pre, post, weight)` columns:

```bash
uv run python3 scripts/simplify_contactome_sqlite_db.py \
    /path/to/contactome.db \
    --output-db /path/to/contactome_simplified.db

```


Or in-place (saving a backup copy as a renamed table):
```bash
uv run python3 scripts/simplify_contactome_sqlite_db.py \
	/path/to/contactome.db \
	--in-place --backup
```

Omit `--in-place` to write a new `*_simplified.db` alongside the source file,
or pass `--output-db /tmp/contactome_simple.db --force` to control the output
path explicitly.

## Merge sqlite db shards

If you have a set of sharded sqlite db files (e.g. produced by workers with
`--shard-sqlite`), you can merge them into a single sqlite db file using:

```bash
uv run scripts/merge_sqlite_dbs.py --out /path/to/merged.db --dir /path/to/sharded/dbs/
```

This will scan the specified directory for all `.db` files and merge the components into a single output database.

