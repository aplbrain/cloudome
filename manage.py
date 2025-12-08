import csv
from collections import defaultdict
import json
import boto3
import networkx as nx
from io import TextIOWrapper
from tqdm.auto import tqdm
import argparse
import csv
from tqdm import tqdm
import re

from database import (
    SynapseEdgeResultsModel,
    ContactomeEdgeTaskPayload,
    ContactEdgeResultsModel,
    TaskType,
    VolumeTaskPayload,
)
from shared_cuboid_utils import generate_cuboidwise_tasks
from shared_utils import (
    _attach_contactome_parser,
    _attach_global_arguments,
    _attach_synapse_parser,
    _parse_mip_argument,
)
from shared_synapse_utils import (
    export_synapse_mask_centroids_to_file,
    generate_centroidwise_tasks,
)


sqs = boto3.client("sqs", region_name="us-east-1")


def enqueue_centroids_from_file(
    sqs_url: str,
    graph_id: str,
    filename: str,
    synapse_channel: str,
    segmentation_channel: str,
    mip: list | int,
    enqueue_limit: int = None,
):
    """
    Read centroids from a file and enqueue them to SQS for processing.
    """
    for i, payload in enumerate(
        tqdm(
            generate_centroidwise_tasks(
                graph_id=graph_id,
                filename=filename,
                synapse_channel=synapse_channel,
                segmentation_channel=segmentation_channel,
                mip=mip,
                enqueue_limit=enqueue_limit,
            )
        )
    ):
        sqs.send_message(QueueUrl=sqs_url, MessageBody=json.dumps(payload))


def generate_cuboidwise_tasks_for_contactome_or_volume(
    sqs_url: str,
    graph_id: str,
    task_type: TaskType,
    segmentation_channel: str,
    mip: list | int,
    block_size: tuple = (64, 64, 64),
    z_start: int = None,
    z_end: int = None,
    enqueue_limit: int = None,
):
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
        sqs.send_message(QueueUrl=sqs_url, MessageBody=json.dumps(payload))


def local_dequeue(sqs_url: str):
    import cloudome

    response = sqs.receive_message(QueueUrl=sqs_url, MaxNumberOfMessages=1)
    if "Messages" in response:
        for message in response["Messages"]:
            cloudome.process_queue_job({"Records": [message]}, None)
            sqs.delete_message(QueueUrl=sqs_url, ReceiptHandle=message["ReceiptHandle"])


def initialize_resources():
    # TODO: Also provision SQS at some point...
    SynapseEdgeResultsModel.create_table(
        billing_mode="PAY_PER_REQUEST",
    )


def export_dynamodb_results_to_csv(graph_id: str, output_file: str):
    """
    Export results for a given graph_id to a CSV file.
    This function streams the results to handle large datasets efficiently.
    """
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["graph_id", "synapse_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # # Stream results from SynapseEdgeResultsModel
        # for result in SynapseEdgeResultsModel.query(graph_id):
        #     writer.writerow({'graph_id': result.graph_id, 'synapse_id': result.synapse_id})

        # Stream results from ContactEdgeResultsModel
        for result in ContactEdgeResultsModel.query(graph_id):
            writer.writerow(
                {"graph_id": result.graph_id, "synapse_id": result.synapse_id}
            )


def simplify_contactome_data(instream: TextIOWrapper, outstream: TextIOWrapper):
    """
    Simplify contactome data by aggregating weights for each (pre, post) pair.
    """
    # Use a dictionary to accumulate weights for each (pre, post) pair
    weights = defaultdict(int)

    reader = csv.reader(instream)
    writer = csv.writer(outstream)

    # Read and process each row
    for row in reader:
        # Extract pre, post, and weight from the synapse_id field
        match = re.search(r"pre(\d+)_post(\d+)_w(\d+)", row[1])
        if match:
            pre, post, weight = match.groups()
            key = (pre, post)
            weights[key] += int(weight)

    # Write the aggregated results to the output stream
    writer.writerow(["id1", "id2", "nm^2"])  # Header
    for (pre, post), total_weight in weights.items():
        writer.writerow([pre, post, total_weight])


def simplify_volume_data(instream: TextIOWrapper, outstream: TextIOWrapper):
    """
    Simplify volume data by summing voxel counts per seg ID
    """
    # Use a dictionary to accumulate voxel counts for each seg ID
    voxel_counts = defaultdict(int)

    reader = csv.reader(instream)
    writer = csv.writer(outstream)

    # Read and process each row
    for row in reader:
        # The row looks like a tuple of (dataset ID, "vol_x100_y20_z42_seg19934_v1263")
        match = re.search(r"seg(\d+)_v(\d+)", row[1])
        if match:
            seg_id, voxel_count = match.groups()
            voxel_counts[seg_id] += int(voxel_count)

    # Write the aggregated results to the output stream
    writer.writerow(["seg_id", "voxel_count"])  # Header
    for seg_id, total_count in voxel_counts.items():
        writer.writerow([seg_id, total_count])


def simplify_synapse_data(
    raw_file: TextIOWrapper, output_file: str, invalid_nodes: list, simple: bool = False
):
    """
    Simplify synapse data by processing raw export and generating an edgelist CSV.
    """
    # Create a directed multigraph
    g = nx.MultiDiGraph()

    # Skip header
    next(raw_file)
    for line in raw_file:
        if line.startswith("#"):
            continue
        _, edge_raw = line.strip().split(",")
        # Parse synapse data (e.g., syn_x1000_y1068_z444_pre-1_post-1)
        _, x, y, z, pre, post = edge_raw.split("_")
        x, y, z = int(x[1:]), int(y[1:]), int(z[1:])
        pre = pre[len("pre") :]
        post = post[len("post") :]
        g.add_edge(pre, post, pos=(x, y, z))

    # Remove invalid nodes (-1 and 0)
    g.remove_nodes_from(invalid_nodes)

    # If simple is True, downcast to a simple graph
    if simple:
        g = nx.DiGraph(g)

    # Save the simplified graph as an edgelist
    nx.write_edgelist(g, output_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cloudome Command Line Interface")

    # Global arguments for multiple commands
    _attach_global_arguments(parser)
    subparsers = parser.add_subparsers(dest="namespace", required=True)

    # Namespace: synapses
    (
        synapses_parser,
        synapses_subparsers,
        (synapses_generate_parser, enqueue_parser),
    ) = _attach_synapse_parser(subparsers)

    # Subcommand: simplify (synapses)
    syn_simplify_parser = synapses_subparsers.add_parser(
        "simplify", help="Simplify raw synapse data to a CSV file"
    )
    syn_simplify_parser.add_argument(
        "--raw-file",
        type=str,
        required=True,
        help="Path to CSV file with raw exported synapse data (from `export` command)",
    )
    syn_simplify_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path for simplified synapse data",
    )
    syn_simplify_parser.add_argument(
        "--invalid-nodes",
        type=str,
        nargs="*",
        default=["-1", "0"],
        help="List of invalid nodes to remove from the graph",
    )
    syn_simplify_parser.add_argument(
        "--simple", action="store_true", help="If set, downcast to a simple graph"
    )

    # Namespace: contactome
    (contactome_parser, contactome_subparsers, (contactome_generate_parser,)) = (
        _attach_contactome_parser(subparsers)
    )

    # Subcommand: simplify (contactome)
    contactome_simplify_parser = contactome_subparsers.add_parser(
        "simplify", help="Simplify contactome raw export to edgelist CSV"
    )
    contactome_simplify_parser.add_argument(
        "--raw-file",
        type=str,
        required=True,
        help="Path to CSV file with raw exported contactome data (from `export` command)",
    )
    contactome_simplify_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path for simplified contactome data",
    )

    # Namespace: volume
    volume_parser = subparsers.add_parser(
        "volume", help="Commands related to volume computation"
    )
    volume_subparsers = volume_parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate (volume)
    volume_generate_parser = volume_subparsers.add_parser(
        "generate", help="Generate cuboidwise tasks for volume"
    )
    volume_generate_parser.add_argument(
        "--graph-id", type=str, required=True, help="Graph ID for processing"
    )
    volume_generate_parser.add_argument(
        "--segmentation-channel",
        type=str,
        required=True,
        help="S3 path to segmentation channel data",
    )
    volume_generate_parser.add_argument(
        "--block-size-x", type=int, default=64, help="Block size for X dimension"
    )
    volume_generate_parser.add_argument(
        "--block-size-y", type=int, default=64, help="Block size for Y dimension"
    )
    volume_generate_parser.add_argument(
        "--block-size-z", type=int, default=32, help="Block size for Z dimension"
    )
    volume_generate_parser.add_argument(
        "--z-start", type=int, default=None, help="Starting Z slice"
    )
    volume_generate_parser.add_argument(
        "--z-end", type=int, default=None, help="Ending Z slice"
    )
    volume_generate_parser.add_argument(
        "--enqueue-limit",
        type=int,
        default=None,
        help="Limit the number of tasks to enqueue",
    )

    # Subcommand: simplify (volume)
    volume_simplify_parser = volume_subparsers.add_parser(
        "simplify", help="Simplify volume raw export to edgelist CSV"
    )
    volume_simplify_parser.add_argument(
        "--raw-file",
        type=str,
        required=True,
        help="Path to CSV file with raw exported volume data (from `export` command)",
    )
    volume_simplify_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path for simplified volume data",
    )

    # Namespace: export
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument(
        "graph_id", type=str, help="Graph ID to export results for"
    )
    export_parser.add_argument("output_file", type=str, help="Output CSV file path")

    # Namespace: dequeue
    dequeue_parser = subparsers.add_parser(
        "dequeue", help="Process a single job from the queue"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    mip = _parse_mip_argument(args.mip)

    if args.namespace == "synapses":
        if args.command == "generate":
            export_synapse_mask_centroids_to_file(
                synapse_channel=args.synapse_channel,
                output_file=args.output_file,
                mip=mip,
            )
        elif args.command == "enqueue":
            enqueue_centroids_from_file(
                sqs_url=args.queue_url,
                graph_id=args.graph_id,
                filename=args.centroids_file,
                synapse_channel=args.synapse_channel,
                segmentation_channel=args.segmentation_channel,
                mip=mip,
                enqueue_limit=args.enqueue_limit,
            )
        elif args.command == "simplify":
            with open(args.raw_file, "r") as infile:
                simplify_synapse_data(
                    infile,
                    output_file=args.output_file,
                    invalid_nodes=args.invalid_nodes,
                    simple=args.simple,
                )
    elif args.namespace == "contactome":
        if args.command == "generate":
            block_size = (args.block_size_x, args.block_size_y, args.block_size_z)
            generate_cuboidwise_tasks_for_contactome_or_volume(
                sqs_url=args.queue_url,
                graph_id=args.graph_id,
                task_type="contactome",
                segmentation_channel=args.segmentation_channel,
                mip=mip,
                block_size=block_size,
                z_start=args.z_start,
                z_end=args.z_end,
                enqueue_limit=args.enqueue_limit,
            )
        elif args.command == "simplify":
            with (
                open(args.raw_file, "r") as infile,
                open(args.output_file, "w") as outfile,
            ):
                simplify_contactome_data(instream=infile, outstream=outfile)
    elif args.namespace == "volume":
        if args.command == "generate":
            block_size = (args.block_size_x, args.block_size_y, args.block_size_z)
            generate_cuboidwise_tasks_for_contactome_or_volume(
                sqs_url=args.queue_url,
                graph_id=args.graph_id,
                task_type="volume",
                segmentation_channel=args.segmentation_channel,
                mip=mip,
                block_size=block_size,
                z_start=args.z_start,
                z_end=args.z_end,
                enqueue_limit=args.enqueue_limit,
            )
        elif args.command == "simplify":
            with (
                open(args.raw_file, "r") as infile,
                open(args.output_file, "w") as outfile,
            ):
                simplify_volume_data(instream=infile, outstream=outfile)
    elif args.namespace == "export":
        export_dynamodb_results_to_csv(args.graph_id, args.output_file)
    elif args.namespace == "dequeue":
        local_dequeue(args.queue_url)


if __name__ == "__main__":
    main()
