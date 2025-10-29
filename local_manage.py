import argparse
from functools import partial

from database import ContactomeEdgeTaskPayload, TaskType, VolumeTaskPayload
from tqdm import tqdm
from taskqueue import TaskQueue, queueable
from shared_utils import generate_cuboidwise_tasks

QUEUE_LEASE_SECONDS = 60 * 5  # 5 minutes


@queueable
def _process_contactome_task(task_payload: ContactomeEdgeTaskPayload):
    print(f"Processing contactome task: {task_payload}")


def generate_cuboidwise_tasks_for_contactome_or_volume(
    fq_url: str,
    graph_id: str,
    task_type: TaskType,
    segmentation_channel: str,
    mip: list | int,
    block_size: tuple = (64, 64, 64),
    z_start: int = None,
    z_end: int = None,
    enqueue_limit: int = None,
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
        tq.insert(partial(_process_contactome_task, task_payload=payload))


def run_contactome_worker(fq_url: str, verbose: bool = False, tally: bool = True):
    tq = TaskQueue(fq_url)
    tq.poll(lease_seconds=QUEUE_LEASE_SECONDS, verbose=verbose, tally=tally)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cloudome Command Line Interface")

    # Global arguments for multiple commands
    parser.add_argument(
        "--mip",
        type=str,
        default="72,72,84",
        help="MIP value as either a single int or comma-separated values (e.g., 72,72,84)",
    )
    parser.add_argument(
        "--fq-url",
        type=str,
        default="fq://CloudomeJobs",
        help="FQ URL for job queue",
    )

    subparsers = parser.add_subparsers(dest="namespace", required=True)

    # Namespace contactome
    contactome_parser = subparsers.add_parser(
        "contactome", help="Commands related to contactome generation"
    )
    contactome_subparsers = contactome_parser.add_subparsers(
        dest="command", required=True
    )
    contactome_subparsers.add_parser("generate", help="Generate contactome tasks")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.namespace == "contactome":
        if args.command == "generate":
            block_size = (args.block_size_x, args.block_size_y, args.block_size_z)
            generate_cuboidwise_tasks_for_contactome_or_volume(
                sqs_url=args.sqs_url,
                graph_id=args.graph_id,
                task_type="contactome",
                segmentation_channel=args.segmentation_channel,
                mip=mip,
                block_size=block_size,
                z_start=args.z_start,
                z_end=args.z_end,
                enqueue_limit=args.enqueue_limit,
            )
            generate_contactome_tasks(args)
