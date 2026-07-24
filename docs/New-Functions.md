# Adding a New Task Type

This is the practical path to introduce a new task type that runs locally via cloudome.

## Tasks involve:

-   A distinct payload shape (`TypedDict`) with required fields.
-   A generator that produces those payloads (shared utils).
-   A local worker handler (`@queueable`) that processes one payload.
-   CLI wiring to enqueue payloads and run workers.
-   Optional persistence: SQLite tables (local) or DynamoDB items (cloud)

## Shared Utils: Generators and Helpers

-   Generators live under `src/cloudome/shared_*_utils.py`. They yield `TypedDict` payloads.
-   Examples:
    -   `generate_cuboidwise_tasks` in `src/cloudome/shared_cuboid_utils.py` produces `ContactomeEdgeTaskPayload | VolumeTaskPayload` from a segmentation source and block layout.
    -   `generate_centroidwise_tasks` in `src/cloudome/shared_synapse_utils.py` reads centroid CSVs and yields `SynapseEdgeTaskPayload`.
-   Keep generators pure: they shouldn’t write to DBs or queues; just yield payloads.

## Define the Payload and (Optionally) Results Model

-   In `src/cloudome/database.py`, add a new `TypedDict` for your task, e.g.:
    -   `class MyTaskPayload(TypedDict): ...`
-   For cloud persistence, define a DynamoDB model (like `SynapseEdgeResultsModel`) and decide on a human-inspectable `range_key` format.

## Local Worker Handler (`src/cloudome/local_manage.py`)

-   Add a `@queueable` function that:
    -   Resolves the sharded SQLite path with `_resolve_sqlite_path`.
    -   Validates payload assumptions (bounds, shapes, required fields).
    -   Creates the destination table with a stable primary key.
    -   Inserts rows derived from your computation.
-   Follow the existing patterns:
    -   Contactome writes `contactome_edges(graph_id, location, pre, post, weight)` with PK `(graph_id, location, pre, post)`.
    -   Synapses write `synapse_edges(graph_id, pre, post, xyz_loc)` with PK `(graph_id, xyz_loc)`.
    -   Volume writes `volume_counts(graph_id, location, seg_id, voxel_count)` with PK `(graph_id, location, seg_id)`.
-   If your task needs volume/segmentation windowing, reuse CloudVolume bounds helpers from `src/cloudome/cloudome.py` (e.g., `_bbox_min_max`, `return_volume_counts`).

## CLI Wiring

-   Extend `parse_arguments()` in `src/cloudome/local_manage.py`:
    -   Add a new namespace (e.g., `mytask`) and a `generate` subcommand.
    -   Parse flags that your generator needs (channels, block sizes, ranges, limits).
-   In `main()`:
    -   Provision the SQLite table before enqueue.
    -   Route enqueue to your generator and insert queueables via `TaskQueue`.
    -   Reuse `enqueue_cuboidwise_tasks_for_contactome_or_volume` if compatible, or add a small enqueue function mirroring synapses/contactome.

## Cloud Path (Optional)

-   If this task should run in the Lambda flow, update `cloudome.process_queue_job`:
    -   Add a branch detecting your payload and call a compute function.
    -   Save to a DynamoDB model with a predictable `synapse_id`/item key format.
-   Ensure the model table/queue exists (see `src/cloudome/manage.py` `initialize_resources()`).

## SQLite Schema Strategy

-   Use human-readable composite keys so merges and deduplication are straightforward:
    -   Include a `location` string like `ctc_x{xs}_y{ys}_z{zs}` or `vol_x{...}`.
    -   Avoid single surrogate keys; prefer PKs that encode uniqueness naturally.
-   Match patterns already used to keep scripts like `merge_sqlite_dbs.py` schema-agnostic.

## End-to-End: Minimal Steps

1. Payload: add `MyTaskPayload` in `src/cloudome/database.py`.
2. Generator: add `generate_mytask_payloads(...)` under `src/cloudome/shared_*_utils.py`.
3. Worker: implement `@queueable _process_mytask(...)` in `src/cloudome/local_manage.py`.
4. Schema: create a `mytask_*` table with a composite PK that matches how you ensure uniqueness.
5. CLI: add `mytask generate` to enqueue, and use `worker` to process.
6. Test locally: enqueue a few items and run worker with `--dequeue-limit`.

## Try It (Local Queue)

```bash
uv run python local_manage.py mytask generate [flags]
uv run python local_manage.py worker --dequeue-limit 5
```

## Notes on Sharded SQLite

-   Pass `--shard-sqlite` to generate a unique DB file per worker, reducing lock contention.
-   `_resolve_sqlite_path()` derives a shard suffix using host/pid/timestamp and caches per base path.

## Patterns to Copy, Not Re-Invent

-   Compute functions live in `src/cloudome/cloudome.py` and are callable from both cloud and local paths.
-   Location key formats (`ctc_x...`, `syn_x...`, `vol_x...`) are consistent across components.
-   Error handling: raise `ValueError` where inputs are invalid; return safe defaults in cloud paths when necessary.
