import argparse
import datetime
from functools import partial
import os
import socket
import sqlite3
from pathlib import Path

from tqdm import tqdm
from taskqueue import TaskQueue, queueable

from .database import (
    ContactomeEdgeTaskPayload,
    SynapseEdgeTaskPayload,
    TaskType,
    VolumeTaskPayload,
)
from .shared_cuboid_utils import generate_cuboidwise_tasks
from .shared_synapse_utils import (
    generate_centroidwise_tasks,
    export_synapse_mask_centroids_to_file,
)
from .shared_utils import (
    _attach_contactome_parser,
    _attach_global_arguments,
    _attach_synapse_parser,
    _attach_volume_parser,
    _parse_mip_argument,
)
from .cloudome import return_ctc_edges, return_seg_edge, return_volume_counts

QUEUE_LEASE_SECONDS = 60 * 5  # 5 minutes
_SQLITE_SHARDING_ENABLED = False
_SQLITE_SHARD_SUFFIX: str | None = None
_SQLITE_PATH_CACHE: dict[str, str] = {}


@queueable
def _process_contactome_task(
    task_payload: ContactomeEdgeTaskPayload, sqlite_db_path: str
):
    """Thin wrapper to process a contactome task inside a queueable."""
    sqlite_db_path = _resolve_sqlite_path(sqlite_db_path)
    edges = return_ctc_edges(task_payload)
    # save to sqlite db
    conn = sqlite3.connect(sqlite_db_path)
    # We assume the table `contactome_edges` exists with columns:
    # graph_id  # string, like "foo"
    # location  # "ctc_x100_y20_z42"
    # pre       # u64
    # post      # u64
    # weight    # u64
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS contactome_edges (
            graph_id TEXT,
            location TEXT,
            pre TEXT,
            post TEXT,
            weight INTEGER,
            PRIMARY KEY (graph_id, location, pre, post)
        )
        """
    )
    for (seg_a, seg_b), weight in edges.items():
        location_str = f"ctc_x{task_payload['cuboid_start'][0]}_y{task_payload['cuboid_start'][1]}_z{task_payload['cuboid_start'][2]}"
        cursor.execute(
            """
            INSERT INTO contactome_edges (graph_id, location, pre, post, weight)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_payload["graph_id"], location_str, str(seg_a), str(seg_b), weight),
        )
    conn.commit()
    conn.close()


@queueable
def _process_synapse_task(task_payload: SynapseEdgeTaskPayload, sqlite_db_path: str):
    """Thin wrapper to process a synapse task inside a queueable."""
    sqlite_db_path = _resolve_sqlite_path(sqlite_db_path)
    edge = return_seg_edge(task_payload)
    # Assume table `synapse_edges` exists with columns:
    # graph_id  # string, like "foo"
    # pre       # u64
    # post      # u64
    # xyz_loc   # string, like "x,y,z"
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS synapse_edges (
            graph_id TEXT,
            pre TEXT,
            post TEXT,
            xyz_loc TEXT,
            PRIMARY KEY (graph_id, xyz_loc)
        )
        """
    )
    xyz_loc = f"{task_payload['centroid_xyz'][0]},{task_payload['centroid_xyz'][1]},{task_payload['centroid_xyz'][2]}"
    cursor.execute(
        """
        INSERT INTO synapse_edges (graph_id, pre, post, xyz_loc)
        VALUES (?, ?, ?, ?)
        """,
        (task_payload["graph_id"], str(edge[0]), str(edge[1]), xyz_loc),
    )
    conn.commit()
    conn.close()


def provision_db_contactome(sqlite_db_path: str):
    """Provision the local SQLite database."""
    conn = sqlite3.connect(_resolve_sqlite_path(sqlite_db_path))
    cursor = conn.cursor()
    # Create table for contactome edges
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS contactome_edges (
            graph_id TEXT,
            location TEXT,
            pre TEXT,
            post TEXT,
            weight INTEGER,
            PRIMARY KEY (graph_id, location, pre, post)
        )
        """
    )
    conn.commit()
    conn.close()


def provision_db_synapses(sqlite_db_path: str):
    """Provision the local SQLite database for synapse edges."""
    conn = sqlite3.connect(_resolve_sqlite_path(sqlite_db_path))
    cursor = conn.cursor()
    # Create table for synapse edges
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS synapse_edges (
            graph_id TEXT,
            pre TEXT,
            post TEXT,
            xyz_loc TEXT,
            PRIMARY KEY (graph_id, xyz_loc)
        )
        """
    )
    conn.commit()
    conn.close()


def provision_db_volume(sqlite_db_path: str):
    """Provision the local SQLite database for per-seg volume counts."""
    conn = sqlite3.connect(_resolve_sqlite_path(sqlite_db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS volume_counts (
            graph_id TEXT,
            location TEXT,
            seg_id TEXT,
            voxel_count INTEGER,
            PRIMARY KEY (graph_id, location, seg_id)
        )
        """
    )
    conn.commit()
    conn.close()


def enqueue_cuboidwise_tasks_for_contactome_or_volume(
    fq_url: str,
    graph_id: str,
    task_type: TaskType,
    segmentation_channel: str,
    mip: list | int,
    sqlite_db_path: str,
    block_size: tuple = (64, 64, 64),
    z_start: int | None = None,
    z_end: int | None = None,
    enqueue_limit: int | None = None,
):
    tq = TaskQueue(fq_url)
    for task in tqdm(
        generate_cuboidwise_tasks(
            graph_id=graph_id,
            task_type=task_type,
            segmentation_channel=segmentation_channel,
            mip=mip,
            block_size=block_size,
            z_start=z_start,
            z_end=z_end,
            enqueue_limit=enqueue_limit,
        )
    ):
        payload: ContactomeEdgeTaskPayload | VolumeTaskPayload = task
        if task_type == "contactome":
            tq.insert(
                partial(
                    _process_contactome_task,
                    task_payload=payload,  # type: ignore[arg-type]
                    sqlite_db_path=sqlite_db_path,
                )
            )
        else:
            tq.insert(
                partial(
                    _process_volume_task,
                    task_payload=payload,  # type: ignore[arg-type]
                    sqlite_db_path=sqlite_db_path,
                )
            )


@queueable
def _process_volume_task(task_payload: VolumeTaskPayload, sqlite_db_path: str):
    """Process a volume task and persist per-seg voxel counts to SQLite."""
    sqlite_db_path = _resolve_sqlite_path(sqlite_db_path)
    counts = return_volume_counts(task_payload)
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS volume_counts (
            graph_id TEXT,
            location TEXT,
            seg_id TEXT,
            voxel_count INTEGER,
            PRIMARY KEY (graph_id, location, seg_id)
        )
        """
    )
    location_str = f"vol_x{task_payload['cuboid_start'][0]}_y{task_payload['cuboid_start'][1]}_z{task_payload['cuboid_start'][2]}"
    for seg_id, voxels in counts.items():
        cursor.execute(
            """
            INSERT INTO volume_counts (graph_id, location, seg_id, voxel_count)
            VALUES (?, ?, ?, ?)
            """,
            (
                task_payload["graph_id"],
                location_str,
                str(seg_id),
                int(voxels),
            ),
        )
    conn.commit()
    conn.close()


def enqueue_centroids_from_file(
    fq_url: str,
    graph_id: str,
    filename: str,
    synapse_channel: str,
    segmentation_channel: str,
    mip: list | int,
    sqlite_db_path: str,
    enqueue_limit: int = None,
):
    """
    Read centroids from a file and enqueue them to SQS for processing.
    """
    tq = TaskQueue(fq_url)
    for i, payload in enumerate(
        generate_centroidwise_tasks(
            graph_id=graph_id,
            filename=filename,
            synapse_channel=synapse_channel,
            segmentation_channel=segmentation_channel,
            mip=mip,
            enqueue_limit=enqueue_limit,
        )
    ):
        tq.insert(
            partial(
                _process_synapse_task,
                task_payload=payload,
                sqlite_db_path=sqlite_db_path,
            )
        )


def run_worker(
    fq_url: str,
    verbose: bool = False,
    tally: bool = True,
    max_tasks: int | None = None,
):
    tq = TaskQueue(fq_url)
    tq.poll(
        lease_seconds=QUEUE_LEASE_SECONDS,
        verbose=verbose,
        tally=tally,
        stop_fn=(lambda executed: executed >= max_tasks) if max_tasks else None,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cloudome Command Line Interface")
    _attach_global_arguments(parser, queue_url_default="fq://q-CloudomeTasks")
    # SQLite DB path argument
    parser.add_argument(
        "--sqlite-db-path",
        type=str,
        default="cloudome-results.db",
        help="Path to the local SQLite database for task tracking",
    )
    parser.add_argument(
        "--shard-sqlite",
        action="store_true",
        help="Use a worker-specific shard of the SQLite database to reduce lock contention",
    )

    subparsers = parser.add_subparsers(dest="namespace", required=True)
    _attach_contactome_parser(subparsers)
    _attach_synapse_parser(subparsers)
    _attach_volume_parser(subparsers)

    worker_parser = subparsers.add_parser("worker", help="Run a task worker")
    worker_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for the worker",
    )
    worker_parser.add_argument(
        "--tally",
        action="store_true",
        help="Enable task tallying for the worker",
    )
    worker_parser.add_argument(
        "--dequeue-limit",
        type=int,
        default=None,
        help="Maximum number of tasks to process before exiting",
    )

    return parser.parse_args()


def _enable_sqlite_sharding():
    global _SQLITE_SHARDING_ENABLED, _SQLITE_SHARD_SUFFIX
    if _SQLITE_SHARDING_ENABLED:
        return
    host = socket.gethostname().replace(".", "-")
    pid = os.getpid()
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    _SQLITE_SHARD_SUFFIX = f"{host}-{pid}-{timestamp}"
    _SQLITE_SHARDING_ENABLED = True


def _resolve_sqlite_path(sqlite_db_path: str) -> str:
    if not _SQLITE_SHARDING_ENABLED:
        return sqlite_db_path
    cached = _SQLITE_PATH_CACHE.get(sqlite_db_path)
    if cached:
        return cached
    suffix = _SQLITE_SHARD_SUFFIX or "shard"
    path = Path(sqlite_db_path)
    parent = path.parent if path.parent != Path("") else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    if path.suffix:
        resolved = parent / f"{path.stem}.{suffix}{path.suffix}"
    else:
        resolved = parent / f"{path.name}.{suffix}"
    resolved_path = str(resolved)
    _SQLITE_PATH_CACHE[sqlite_db_path] = resolved_path
    return resolved_path


def main():
    args = parse_arguments()
    mip = _parse_mip_argument(args.mip)

    if args.shard_sqlite:
        _enable_sqlite_sharding()

    if args.namespace == "contactome":
        if args.command == "generate":
            block_size = (args.block_size_x, args.block_size_y, args.block_size_z)
            provision_db_contactome(args.sqlite_db_path)
            enqueue_cuboidwise_tasks_for_contactome_or_volume(
                fq_url=args.queue_url,
                graph_id=args.graph_id,
                task_type="contactome",
                segmentation_channel=args.segmentation_channel,
                sqlite_db_path=args.sqlite_db_path,
                mip=mip,
                block_size=block_size,
                z_start=args.z_start,
                z_end=args.z_end,
                enqueue_limit=args.enqueue_limit,
            )

    elif args.namespace == "synapses":
        if args.command == "generate":
            export_synapse_mask_centroids_to_file(
                synapse_channel=args.synapse_channel,
                output_file=args.output_file,
                mip=mip,
            )
        elif args.command == "enqueue":
            provision_db_synapses(args.sqlite_db_path)
            enqueue_centroids_from_file(
                fq_url=args.queue_url,
                graph_id=args.graph_id,
                filename=args.centroids_file,
                sqlite_db_path=args.sqlite_db_path,
                synapse_channel=args.synapse_channel,
                segmentation_channel=args.segmentation_channel,
                mip=mip,
                enqueue_limit=args.enqueue_limit,
            )

    elif args.namespace == "volume":
        if args.command == "generate":
            block_size = (args.block_size_x, args.block_size_y, args.block_size_z)
            provision_db_volume(args.sqlite_db_path)
            enqueue_cuboidwise_tasks_for_contactome_or_volume(
                fq_url=args.queue_url,
                graph_id=args.graph_id,
                task_type="volume",
                segmentation_channel=args.segmentation_channel,
                sqlite_db_path=args.sqlite_db_path,
                mip=mip,
                block_size=block_size,
                z_start=args.z_start,
                z_end=args.z_end,
                enqueue_limit=args.enqueue_limit,
            )

    elif args.namespace == "worker":
        run_worker(
            fq_url=args.queue_url,
            verbose=args.verbose,
            tally=args.tally,
            max_tasks=args.dequeue_limit,
        )


if __name__ == "__main__":
    main()
