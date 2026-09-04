# Local Use

Cloudome uses a producer-consumer pattern. Step 1 is to split the computation into discrete jobs and enqueue them to a task queue, and step 2 is to run workers which will execute those jobs. The final steps are postprocessing.

## 1. Queue jobs for a chosen computation

To generate contactomes, connectomes, and volumes locally, you can use the following commands. This will not deploy any cloud resources (no SQS, no Lambdas).

### Contactomes

Provision the queue with one job per chunk of data:

```bash
uv run local_manage.py \
  --sqlite-db-path test.db
  --mip 4,4,40 \
  --queue-url fq://q-CloudomeTasks \
  contactome generate \
  --graph-id test \
  --segmentation-channel s3://path/to/segmentation \
  --block-size-x 512 --block-size-y 512 --block-size-z 128 \
  --z-start 0 --z-end 1000
```
Parameters:  
**--sqlite-db-path**: The filename of the database which Cloudome will write to.  
**--mip**: Indicates the resolution at which to compute the contactome. We recommend using a downsampled version of segmentation so that runtime is quicker. Downsampling with max-pooling will ensure that segmentation labels are preserved. This will especially be necessary when segmentation does not cover membranes, as downsampling with max-pooling will cause the segmentation labels to crowd out the 0-labeled membranes and produce more contacts.  
**--queue-url**: The disk location to which tasks should be queued.
**--graph-id**: A column in the database that will differentiate runs, so that the same database can be used for multiple runs.  
**--segmentation-channe**l: The path to the segmentation that will be used in the computation. Any protocol supported by CloudVolume is supported (e.g. s3://, gs://, https://, file://, etc)  
**--block-size-&lt;dim&gt;**: These three parameters determine the chunk size of one job. This size should always be multiples of the dataset's chunk size. For best performance, choose a maximal chunk size that can fit into the amount of memory allocated to a single job.  
**--z-start**, **--z-stop**: Optionally limit the Z bounds of the computation.

To clear out stale tasks, run `rm -rf ./q-Cloudome-Tasks` to delete the queue, replacing with your custom location if using the `--queue-url` parameter. We recommend doing this anytime tasks are provisioned incorrectly, to ensure that the incorrect tasks are not executed.
  
### Volume

Provision the queue with one job per chunk of data:

```bash
uv run local_manage.py \
  --sqlite-db-path zebrafish-nuclei-volume.db \
  --mip 72,72,168 \
  volume generate \
  --graph-id zebrafish-nuclei-volume \
  --segmentation-channel s3://path/to/segmentation/ \
  --block-size-x 512 --block-size-y 512 --block-size-z 128 
```

### Connectomes

Generate synapse centroids file:

```bash
uv run local_manage.py \
  synapses generate \
  --synapse-channel s3://path/to/synapses \
  --output-file synapse-centroids-test0.csv
```

Provision the queue with one job per synapse:

```bash
uv run local_manage.py \
  --sqlite-db-path synapse_edges_test0.db \
  synapses enqueue \
  --graph-id connectome-test0 \
  --centroids-file synapse-centroids-test0.csv \
  --synapse-channel s3://path/to/synapses \
  --segmentation-channel s3://path/to/segmentation
```

### Sharding SQLite DB

On highly parallel systems such as shared-filesystem clusters, you will likely want to shard the sqlite DB. Pass `--shard-sqlite` to any command that takes a database path (e.g. `--sqlite-db-path synapse_edges_test0.db --shard-sqlite`). This will create a unique sqlite DB per worker process.

## 2. Run workers

This command will start a single worker to execute the jobs that were just queued. It's recommended to run multiple workers using whatever parallellized approach you prefer. Tmux and Screen are great for local runs while the most popular software for HPC clusters is Slurm.

```bash
uv run local_manage.py worker
```

You can also limit the number of tasks processed by each worker for testing purposes:

```bash
uv run local_manage.py worker --dequeue-limit 10
```

The result will be a SQLite DB (or set of DBs, if sharded) containing the aggregated outputs of each worker.

## 3. Simplify a chunk-by-chunk SQLite database

This step is not necessary for a connectome.

### Contactomes

Once you have a populated contactome SQLite database, you can aggregate it in-place or
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

### Volumes

A similar script is provided for volume. The --inplace and --backup flags will also work here. 

```bash
uv run python3 scripts/simplify_volume_sqlite_db.py \
  /path/to/volume.db  \
  --output-db /path/to/volume_simplified.db
```

## 4. Merge SQLite DB shards

If you have a set of sharded sqlite db files (e.g. produced by workers with
`--shard-sqlite`), you can merge them into a single sqlite db file using:

```bash
uv run scripts/merge_sqlite_dbs.py \
  --out /path/to/merged.db \
  --dir /path/to/sharded/dbs/
```

This will scan the specified directory for all `.db` files and merge the components into a single output database.

## 5. Export as CSV

If your results are of reasonable size, export them as a CSV.

```bash
uv run scripts/download_table_as_csv.py \
  --db-path /path/to/db_simplified.db \
  --csv-path /path/to/db.csv \
  --table_name contactome_edges
```
