import argparse
from intern.utils.parallel import block_compute
from cloudvolume import CloudVolume
from tqdm import tqdm


from database import TaskType


def generate_cuboidwise_tasks(
    graph_id: str,
    task_type: TaskType,
    segmentation_channel: str,
    mip: list | int,
    block_size: tuple = (64, 64, 64),
    z_start: int = None,
    z_end: int = None,
    enqueue_limit: int = None,
):
    # Create a file with each line being a cuboid start and radius
    seg_data = CloudVolume(segmentation_channel, mip=mip, cache=True)

    if z_end:
        z_end = z_end if z_end < int(seg_data.shape[2]) else int(seg_data.shape[2])
    else:
        z_end = int(seg_data.shape[2])

    blocks = block_compute(
        x_start=0,
        x_stop=int(seg_data.shape[0]),
        y_start=0,
        y_stop=int(seg_data.shape[1]),
        z_start=z_start or 0,
        z_stop=z_end,
        block_size=block_size,
    )

    # if enqueue_limit:
    #     print(f"Queueing {min(len(blocks), enqueue_limit)} blocks")
    # else:
    #     print(f"Queueing {len(blocks)} blocks")

    for i, ((x_start, x_stop), (y_start, y_stop), (z_start, z_stop)) in enumerate(
        blocks
    ):
        if enqueue_limit is not None and i >= enqueue_limit:
            break

        # enqueue a ContactomeEdgeTaskPayload
        payload = {
            "graph_id": graph_id,
            "task_type": task_type,
            "cuboid_start": (x_start, y_start, z_start),
            "cuboid_radius": (x_stop - x_start, y_stop - y_start, z_stop - z_start),
            "segmentation_channel": segmentation_channel,
            "mip": mip,
        }  # ContactomeEdgeTaskPayload | VolumeTaskPayload
        yield payload


def _parse_mip_argument(mip_arg: str) -> list | int:
    # Parse MIP from string - always convert to a list for consistency
    if "," in mip_arg:
        mip = [int(x) for x in mip_arg.split(",")]
        return mip
    else:
        try:
            # If a single value, make it a list with the same value for all dimensions
            single_mip = int(mip_arg)
            mip = [single_mip, single_mip, single_mip]
            return mip
        except ValueError:
            print(
                f"Error: MIP value '{mip_arg}' is not valid. Use a single integer or comma-separated integers."
            )
            exit(1)


def _attach_global_arguments(
    parser: argparse.ArgumentParser,
    queue_url_default: str = "https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
):
    parser.add_argument(
        "--mip",
        type=str,
        default="72,72,84",
        help="MIP value as either a single int or comma-separated values (e.g., 72,72,84)",
    )
    parser.add_argument(
        "--queue-url",
        type=str,
        default=queue_url_default,
        help="Queue URL (SQS URL or FQ file path) for job queue",
    )
    return parser


def _attach_contactome_parser(subparsers: argparse._SubParsersAction):
    contactome_parser = subparsers.add_parser(
        "contactome", help="Commands related to contactome"
    )
    contactome_subparsers = contactome_parser.add_subparsers(
        dest="command", required=True
    )
    contactome_generate_parser = contactome_subparsers.add_parser(
        "generate", help="Generate cuboidwise tasks for contactome"
    )
    contactome_generate_parser.add_argument(
        "--graph-id", type=str, required=True, help="Graph ID for processing"
    )
    contactome_generate_parser.add_argument(
        "--segmentation-channel",
        type=str,
        required=True,
        help="S3 path to segmentation channel data",
    )
    contactome_generate_parser.add_argument(
        "--block-size-x", type=int, default=64, help="Block size for X dimension"
    )
    contactome_generate_parser.add_argument(
        "--block-size-y", type=int, default=64, help="Block size for Y dimension"
    )
    contactome_generate_parser.add_argument(
        "--block-size-z", type=int, default=32, help="Block size for Z dimension"
    )
    contactome_generate_parser.add_argument(
        "--z-start", type=int, default=None, help="Starting Z slice"
    )
    contactome_generate_parser.add_argument(
        "--z-end", type=int, default=None, help="Ending Z slice"
    )
    contactome_generate_parser.add_argument(
        "--enqueue-limit",
        type=int,
        default=None,
        help="Limit the number of tasks to enqueue",
    )
    return contactome_parser, contactome_subparsers, contactome_generate_parser


__all__ = [
    "generate_cuboidwise_tasks",
    "_parse_mip_argument",
    "_attach_global_arguments",
    "_attach_contactome_parser",
]
