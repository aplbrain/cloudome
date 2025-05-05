from intern.utils.parallel import block_compute
import json
import boto3
import networkx as nx
from io import TextIOWrapper
from tqdm.auto import tqdm
from cloudvolume import CloudVolume
import cc3d
import argparse
import csv
import os

from database import SynapseEdgeResultsModel, ContactomeEdgeTaskPayload, ContactEdgeResultsModel


sqs = boto3.client('sqs', region_name='us-east-1')


def get_centroids_for_syn_mask(synapse_channel: str, output_file: str, mip: list|int):
    # Lump pre/post (IDs 2 and 1) into a single binary mask:
    binary_syn_mask = (CloudVolume(synapse_channel, mip=mip, cache=True)[..., 0].squeeze() > 0)
    labels_out, N = cc3d.connected_components(binary_syn_mask, return_N=True)
    stats = cc3d.statistics(labels_out)
    with open(output_file, 'w') as fh:
        for syn_centroid in stats['centroids']:
            fh.write(",".join(map(str, map(int, syn_centroid))) + "\n")


def enqueue_centroids_from_file(sqs_url: str, graph_id: str, filename: str, synapse_channel: str, segmentation_channel: str, mip: list|int, enqueue_limit: int = None):
    """
    Read centroids from a file and enqueue them to SQS for processing.
    """
    with open(filename, 'r') as fh:
        for i, line in enumerate(tqdm(fh)):
            if enqueue_limit is not None and i >= enqueue_limit:
                break
            centroid_xyz = tuple(map(int, line.strip().split(',')))
            payload: SynapseEdgeTaskPayload = {
                "graph_id": graph_id,
                "centroid_xyz": centroid_xyz,
                "synapse_channel": synapse_channel,
                "segmentation_channel": segmentation_channel,
                "mip": mip
            }
            sqs.send_message(
                QueueUrl=sqs_url,
                MessageBody=json.dumps(payload)
            )


def generate_cuboidwise_tasks_for_contactome(sqs_url: str, graph_id: str, segmentation_channel: str, mip: list|int,
                                      block_size: tuple = (64, 64, 64), enqueue_limit: int = None):
    # Create a file with each line being a cuboid start and radius
    seg_data = CloudVolume(segmentation_channel, mip=mip, cache=True)
    blocks = block_compute(
        x_start=0,
        x_stop=int(seg_data.shape[0]),
        y_start=0,
        y_stop=int(seg_data.shape[1]),
        z_start=0,
        z_stop=int(seg_data.shape[2]),
        block_size=block_size,
    )
    print(f"Queueing {len(blocks)} blocks")

    for i, ((x_start, x_stop), (y_start, y_stop), (z_start, z_stop)) in enumerate(blocks):
        if enqueue_limit is not None and i >= enqueue_limit:
            break

        # enqueue a ContactomeEdgeTaskPayload
        payload: ContactomeEdgeTaskPayload = {
            "graph_id": graph_id,
            "cuboid_start": (x_start, y_start, z_start),
            "cuboid_radius": (x_stop - x_start, y_stop - y_start, z_stop - z_start),
            "segmentation_channel": segmentation_channel,
            "mip": mip
        }
        sqs.send_message(
            QueueUrl=sqs_url,
            MessageBody=json.dumps(payload)
        )


def local_dequeue(sqs_url: str):
    import cloudome
    response = sqs.receive_message(
        QueueUrl=sqs_url,
        MaxNumberOfMessages=1
    )
    if "Messages" in response:
        for message in response["Messages"]:
            cloudome.process_queue_job({"Records": [message]}, None)
            sqs.delete_message(
                QueueUrl=sqs_url,
                ReceiptHandle=message["ReceiptHandle"]
            )


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
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['graph_id', 'synapse_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # # Stream results from SynapseEdgeResultsModel
        # for result in SynapseEdgeResultsModel.query(graph_id):
        #     writer.writerow({'graph_id': result.graph_id, 'synapse_id': result.synapse_id})

        # Stream results from ContactEdgeResultsModel
        for result in ContactEdgeResultsModel.query(graph_id):
            writer.writerow({'graph_id': result.graph_id, 'synapse_id': result.synapse_id})

def simplify_contactome_data(instream: TextIOWrapper, outstream: TextIOWrapper):
    """
    Simplify contactome data by aggregating weights for each (pre, post) pair.
    """
    import csv
    from collections import defaultdict

    # Use a dictionary to accumulate weights for each (pre, post) pair
    weights = defaultdict(int)

    reader = csv.reader(instream)
    writer = csv.writer(outstream)

    # Read and process each row
    for row in reader:
        # Extract pre, post, and weight from the synapse_id field
        match = re.match(r"pre(\d+)_post(\d+)_w(\d+)", row[1])
        if match:
            pre, post, weight = match.groups()
            key = (pre, post)
            weights[key] += int(weight)

    # Write the aggregated results to the output stream
    writer.writerow(["pre", "post", "weight"])  # Header
    for (pre, post), total_weight in weights.items():
        writer.writerow([pre, post, total_weight])


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cloudome Command Line Interface")

    # Global arguments for multiple commands
    parser.add_argument("--mip", type=str, default="72,72,84",
                      help="MIP value as either a single int or comma-separated values (e.g., 72,72,84)")
    parser.add_argument("--sqs-url", type=str, default="https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
                        help="SQS URL for job queue")

    subparsers = parser.add_subparsers(dest="namespace", required=True)

    # Namespace: synapses
    synapses_parser = subparsers.add_parser("synapses", help="Commands related to synapses")
    synapses_subparsers = synapses_parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate (synapses)
    syn_generate_parser = synapses_subparsers.add_parser("generate", help="Generate centroids for synapse mask")
    syn_generate_parser.add_argument("--synapse-channel", type=str, required=True,
                                    help="S3 path to synapse channel data")
    syn_generate_parser.add_argument("--output-file", type=str, default="centroids.csv",
                                    help="Output file path for centroids")

    # Subcommand: enqueue (synapses)
    enqueue_parser = synapses_subparsers.add_parser("enqueue", help="Enqueue centroids from file")
    enqueue_parser.add_argument("--graph-id", type=str, required=True,
                              help="Graph ID for processing")
    enqueue_parser.add_argument("--centroids-file", type=str, required=True,
                              help="Path to centroids file")
    enqueue_parser.add_argument("--synapse-channel", type=str, required=True,
                              help="S3 path to synapse channel data")
    enqueue_parser.add_argument("--segmentation-channel", type=str, required=True,
                              help="S3 path to segmentation channel data")
    enqueue_parser.add_argument("--enqueue-limit", type=int, default=None,
                              help="Limit the number of centroids to enqueue")

    # Namespace: contactome
    contactome_parser = subparsers.add_parser("contactome", help="Commands related to contactome")
    contactome_subparsers = contactome_parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate (contactome)
    contactome_generate_parser = contactome_subparsers.add_parser("generate", help="Generate cuboidwise tasks for contactome")
    contactome_generate_parser.add_argument("--graph-id", type=str, required=True,
                                         help="Graph ID for processing")
    contactome_generate_parser.add_argument("--segmentation-channel", type=str, required=True,
                                         help="S3 path to segmentation channel data")
    contactome_generate_parser.add_argument("--block-size-x", type=int, default=64,
                                         help="Block size for X dimension")
    contactome_generate_parser.add_argument("--block-size-y", type=int, default=64,
                                         help="Block size for Y dimension")
    contactome_generate_parser.add_argument("--block-size-z", type=int, default=64,
                                         help="Block size for Z dimension")
    contactome_generate_parser.add_argument("--enqueue-limit", type=int, default=None,
                                         help="Limit the number of tasks to enqueue")

    # Subcommand: simplify (contactome)
    contactome_simplify_parser = contactome_subparsers.add_parser("simplify", help="Simplify contactome raw export to edgelist CSV")
    contactome_simplify_parser.add_argument("--raw-file", type=str, required=True,
                                         help="Path to CSV file with raw exported contactome data (from `export` command)")
    contactome_simplify_parser.add_argument("--output-file", type=str, required=True,
                                         help="Output file path for simplified contactome data")

    # Namespace: export
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument("graph_id", type=str, help="Graph ID to export results for")
    export_parser.add_argument("output_file", type=str, help="Output CSV file path")

    # Namespace: dequeue
    dequeue_parser = subparsers.add_parser("dequeue", help="Process a single job from the queue")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Parse MIP from string - always convert to a list for consistency
    if "," in args.mip:
        mip = [int(x) for x in args.mip.split(",")]
    else:
        try:
            # If a single value, make it a list with the same value for all dimensions
            single_mip = int(args.mip)
            mip = [single_mip, single_mip, single_mip]
        except ValueError:
            print(f"Error: MIP value '{args.mip}' is not valid. Use a single integer or comma-separated integers.")
            exit(1)

    if args.namespace == "synapses":
        if args.command == "generate":
            get_centroids_for_syn_mask(
                synapse_channel=args.synapse_channel,
                output_file=args.output_file,
                mip=mip
            )
        elif args.command == "enqueue":
            enqueue_centroids_from_file(
                sqs_url=args.sqs_url,
                graph_id=args.graph_id,
                filename=args.centroids_file,
                synapse_channel=args.synapse_channel,
                segmentation_channel=args.segmentation_channel,
                mip=mip,
                enqueue_limit=args.enqueue_limit
            )
    elif args.namespace == "contactome":
        if args.command == "generate":
            block_size = (args.block_size_x, args.block_size_y, args.block_size_z)
            generate_cuboidwise_tasks_for_contactome(
                sqs_url=args.sqs_url,
                graph_id=args.graph_id,
                segmentation_channel=args.segmentation_channel,
                mip=mip,
                block_size=block_size,
                enqueue_limit=args.enqueue_limit
            )
        elif args.command == "simplify":
            with open(args.raw_file, 'r') as infile, open(args.output_file, 'w') as outfile:
                simplify_contactome_data(
                    instream=infile,
                    outstream=outfile
                )
    elif args.namespace == "export":
        export_dynamodb_results_to_csv(args.graph_id, args.output_file)
    elif args.namespace == "dequeue":
        local_dequeue(args.sqs_url)


if __name__ == "__main__":
    main()