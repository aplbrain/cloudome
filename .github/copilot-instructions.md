# Cloudome: AI Agent Coding Instructions

## Project Overview
Cloudome is a distributed neuroimaging pipeline that processes 3D volumetric data to extract connectome/contactome information, compute segment volumes, and generate supervoxels. It uses AWS Lambda + SQS for cloud execution and task-queue + SQLite for local development.

## Architecture: Multi-Task Processing Model

**Four main task types** with distinct workflows:

1. **Connectome** (synapse-centroids): Find pre/post segment pairs at synapse locations
   - Task source: `generate_centroidwise_tasks()` reads centroids from CSV
   - Core logic: `return_seg_edge()` in `cloudome.py`
   - Output: Graph edges stored in DynamoDB

2. **Contactome** (cuboid-based): Count contact voxels between all segment pairs per cuboid
   - Task source: `generate_cuboidwise_tasks()` tiles volume by block_size
   - Core logic: `return_ctc_edges()` in `cloudome.py`
   - Output: Weighted edges in DynamoDB

3. **Volume** (cuboid-based): Count total voxels per segment per cuboid
   - Task source: `generate_cuboidwise_tasks()` tiles volume
   - Core logic: `return_volume_counts()` in `cloudome.py`
   - Output: Volume counts in DynamoDB

4. **Supervoxel** (chunk-based watershed): Split segments into supervoxels using edge-guided watershed
   - Task source: `generate_supervoxel_tasks()` tiles volume + generates global IDs
   - Core logic: `return_supervoxel_results()` in `cloudome.py` (orchestrates chunk processing via `shared_supervoxel_utils`)
   - Output: Supervoxel data written directly to output CloudVolume layer (chunk-aligned, no locking)
   - Metadata: Per-chunk stats (num_sv, parent_segment→count mapping, supervoxel sizes) stored in DynamoDB/SQLite

**Data flow**: Task payloads → Queue (SQS/file-queue) → Worker processes → Results → Output layer (supervoxels) + Metadata storage

## Key Files & Responsibilities

| File | Purpose |
|------|---------|
| `cloudome.py` | Core compute logic: `return_seg_edge()`, `return_ctc_edges()`, `return_volume_counts()`, `return_supervoxel_results()` |
| `database.py` | TypedDict payloads (SynapseEdgeTaskPayload, ContactomeEdgeTaskPayload, VolumeTaskPayload, SupervoxelTaskPayload), DynamoDB models |
| `manage.py` | Cloud CLI for AWS SQS enqueue/export |
| `local_manage.py` | Local CLI using file-queue (FileSystemQueue) + SQLite storage |
| `shared_cuboid_utils.py` | Cuboid tiling generator using intern.utils.parallel.block_compute |
| `shared_synapse_utils.py` | Centroid CSV I/O, synapse connected-components labeling |
| `shared_supervoxel_utils.py` | **NEW**: Supervoxel pipeline (GlobalIDPacker, chunk iteration, watershed splitting, task generation) |
| `shared_utils.py` | CLI parser builders (MIP argument parsing, shared flags, supervoxel-specific parsers) |

## Critical Patterns & Conventions

### MIP (Resolution) Handling
- MIP can be scalar `int` or list `[x, y, z]` floats (e.g., `[72, 72, 84]`)
- Always normalize via `_parse_mip_argument()` → returns list
- CloudVolume queries use `mip=cast(Any, mip)` (type casting workaround)

### Bounding Box & Voxel Offset
- CloudVolume has `voxel_offset` (top-left corner) + `shape` (dimensions)
- Always clamp coordinates: `max(min_bound, min(max_bound, coord))`
- Use `_bbox_min_max()` utility to extract minpt/maxpt

### Task Payload Structure
All payloads include `graph_id` (str) + task-specific fields (see `database.py` TypedDicts)
- Cuboid tasks: use `cuboid_start` (origin) + `cuboid_radius` (dimensions)
- Centroid tasks: use `centroid_xyz` tuple
- **Supervoxel tasks**: include `chunk_index_xyz`, `n_chunks_xyz` (for GlobalIDPacker), channel references (seg, raw, output)

### Queue & Results Storage
- **Cloud**: SQS for jobs, DynamoDB (CloudomeResults table) for results
  - Partition key: `graph_id`, Sort key: `synapse_id` (format varies by type)
  - Supervoxel format: `sv_x{cx}_y{cy}_z{cz}_n{num_sv}_p{parent_counts_encoded}`
- **Local**: FileSystemQueue (via taskqueue library), SQLite with per-table schemas
  - Must call `provision_db_*()` before processing to create schema
  - Supervoxel table: `supervoxel_chunks` (stores num_sv, parent_counts, sv_sizes as JSON)

### Supervoxel-Specific Patterns
- **Global ID packing**: `GlobalIDPacker` encodes (chunk_x, chunk_y, chunk_z, local_id) → uint64
  - Ensures no collisions across chunks; unpack with `.decode(gid)` to recover chunk indices
  - Bit layout: [cx | cy | cz | local_id] (most to least significant)
- **Chunk processing with halo**: `iter_chunk_bounds()` yields (chunk_index, write_slices, read_slices)
  - Write region is hard boundary (no halo)
  - Read region includes halo for seamless boundaries between chunks
  - Halo enables contact affinities between adjacent supervoxels across chunk edges
- **Watershed splitting**: Uses edge-guided watershed from raw EM channel
  - Cost = `edge_weight * normalized_edge_cost + (1-edge_weight) * normalized_inverted_distance`
  - Splits based on `target_voxels_per_sv`; merges tiny regions (< `min_voxels_per_sv`)
- **Parent tracking**: Each chunk records which parent segments generated supervoxels
  - Used for downstream affinity computation and agglomeration workflows

### Connected Components & Contact Voxel Counting
- Use `cc3d.connected_components()` for segmentation labeling
- PRESYNAPTIC = 2, POSTSYNAPTIC = 1 (mask values in synapse data)
- `count_contact_voxels()` returns dict keyed by (pre_seg, post_seg) tuple
- Overlapping contacts are deduplicated via `remove_contact_overlap()`

## Development & Testing

### Local Testing (Recommended)
```bash
# Run supervoxel test
uv run python3 -m pytest test_local.py::test_enqueue_supervoxel_tasks

# Or directly:
cd /path/to/cloudome
python3 local_manage.py supervoxel generate \
  --graph-id test-sv \
  --segmentation-channel s3://... \
  --output-channel s3://...output/ \
  --raw-channel s3://...raw/ \
  --chunk-size-x 128 --chunk-size-y 128 --chunk-size-z 128 \
  --z-start 0 --z-end 100

python3 local_manage.py supervoxel run_worker  # or: worker
```
bash
# Run tests with local file-queue + SQLite
uv run python3 -m pytest test_local.py::test_enqueue_contactome_tasks

# Or directly:
cd /path/to/cloudome
python3 local_manage.py contactome generate --graph-id test-ctc --segmentation-channel s3://... --block-size-x 64 --block-size-y 64 --block-size-z 32
python3 local_manage.py contactome run_worker  # polls queue, runs tasks
```

### Cloud Testing
```bash
# Deploys Lambda via Zappa
uv run zappa deploy    # first deploy
uv run zappa update    # incremental updates
uv run zappa schedule  # sets up CloudWatch triggers

# Enqueue via SQS
python3 manage.py contactome generate --graph-id test-ctc --segmentation-channel s3://...
```

### Key Test Parameters
- See `test_local.py` for example: block_size, z_start/z_end, enqueue_limit
- Expected output line counts are hardcoded in tests (update if logic changes)

## Common Tasks & Patterns

### Adding a New Task Type
1. Add `TaskType` literal to `database.py`
2. Create TypedDict payload struct
3. Create `generate_*_tasks()` in `shared_*_utils.py`
4. Implement `return_*_results()` in `cloudome.py`
5. Add `_process_*_task()` wrapper in `local_manage.py`
6. Update CLI parsers in `shared_utils.py` (`_attach_*_parser()`)

### Debugging CloudVolume Issues
- Ensure `fill_missing=True` if data may have gaps
- Check voxel_offset is respected (offsets in absolute space, not relative)
- Bounding box intersection must be non-empty: `x_min < x_max and y_min < y_max and z_min < z_max`
- Use `CLOUD_VOLUME_DIR` env var to control caching location

### Working with SQLite Sharding (Local)
- Enable via `_enable_sqlite_sharding()` in `local_manage.py`
- Uses `SQLITE_SHARD_SUFFIX` to create per-worker DB files
- Merge shards with `scripts/merge_sqlite_dbs.py`

## External Dependencies
- **AWS**: boto3 (SQS), pynamodb (DynamoDB)
- **Data I/O**: CloudVolume, Intern (block_compute)
- **Processing**: numpy, cc3d (connected components), compresso
- **Distribution**: Flask (Lambda entry point), Zappa (serverless deployment), task-queue (job queue abstraction)

## Workspace Configuration
- **Python**: ≥3.11 (requires modern type hints)
- **Package manager**: uv (see pyproject.toml)
- **Deployment**: Zappa config in zappa_settings.json (region=us-east-1, S3 bucket hardcoded in code)
