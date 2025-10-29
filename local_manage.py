import argparse
from functools import partial

from database import ContactomeEdgeTaskPayload, TaskType, VolumeTaskPayload
from tqdm import tqdm
from taskqueue import TaskQueue, queueable
from shared_utils import (
    _attach_contactome_parser,
    _attach_global_arguments,
    _parse_mip_argument,
    generate_cuboidwise_tasks,
)
from cloudome import return_ctc_edges
import sqlite3

QUEUE_LEASE_SECONDS = 60 * 5  # 5 minutes


@queueable
def _process_contactome_task(
    task_payload: ContactomeEdgeTaskPayload, sqlite_db_path: str
):
    """Thin wrapper to process a contactome task inside a queueable."""
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
    for (seg_a, seg_b), weight in edges.items():
        location_str = f"ctc_x{task_payload['cuboid_start'][0]}_y{task_payload['cuboid_start'][1]}_z{task_payload['cuboid_start'][2]}"
        cursor.execute(
            """
            INSERT INTO contactome_edges (graph_id, location, pre, post, weight)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_payload["graph_id"], location_str, seg_a, seg_b, weight),
        )
    conn.commit()
    conn.close()


def provision(sqlite_db_path: str):
    """Provision the local SQLite database."""
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    # Create table for contactome edges
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS contactome_edges (
            graph_id TEXT,
            location TEXT,
            pre INTEGER,
            post INTEGER,
            weight INTEGER,
            PRIMARY KEY (graph_id, location, pre, post)
        )
        """
    )
    conn.commit()
    conn.close()


def generate_cuboidwise_tasks_for_contactome_or_volume(
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
        tq.insert(
            partial(
                _process_contactome_task,
                task_payload=payload,
                sqlite_db_path=sqlite_db_path,
            )
        )


def run_contactome_worker(
    fq_url: str, verbose: bool = False, tally: bool = True, max_tasks: int | None = None
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

    subparsers = parser.add_subparsers(dest="namespace", required=True)
    _attach_contactome_parser(subparsers)

    # Support `worker --jobs N` to run a worker for a specific namespace
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
        "--jobs",
        type=int,
        default=None,
        help="Maximum number of tasks to process before exiting",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    args.mip = _parse_mip_argument(args.mip)

    if args.namespace == "contactome":
        if args.command == "generate":
            block_size = (args.block_size_x, args.block_size_y, args.block_size_z)
            provision(args.sqlite_db_path)
            generate_cuboidwise_tasks_for_contactome_or_volume(
                fq_url=args.queue_url,
                graph_id=args.graph_id,
                task_type="contactome",
                segmentation_channel=args.segmentation_channel,
                sqlite_db_path=args.sqlite_db_path,
                mip=args.mip,
                block_size=block_size,
                z_start=args.z_start,
                z_end=args.z_end,
                enqueue_limit=args.enqueue_limit,
            )
    elif args.namespace == "worker":
        run_contactome_worker(
            fq_url=args.queue_url,
            verbose=args.verbose,
            tally=args.tally,
            max_tasks=args.jobs,
        )


if __name__ == "__main__":
    main()
