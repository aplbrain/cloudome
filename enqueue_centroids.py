from intern.utils.parallel import block_compute
import json
import boto3
from tqdm.auto import tqdm
from cloudvolume import CloudVolume
import cc3d
import argparse
import csv

from database import SynapseEdgeResultsModel, ContactomeEdgeTaskPayload, ContactEdgeResultsModel


sqs = boto3.client('sqs', region_name='us-east-1')

MIP = [72, 72, 84]


def get_centroids_for_syn_mask(synapse_channel: str, mip: list|int):
    binary_syn_mask = (CloudVolume(synapse_channel, mip=mip, cache=True)[..., 0].squeeze() > 0)
    labels_out, N = cc3d.connected_components(binary_syn_mask, return_N=True)
    stats = cc3d.statistics(labels_out)
    with open("../centroids.csv", 'w') as fh:
        for syn_centroid in stats['centroids']:
            fh.write(",".join(map(str, map(int, syn_centroid))) + "\n")


def enqueue_centroids_from_file(sqs_url: str, graph_id: str, filename: str, synapse_channel: str, segmentation_channel: str, mip: list, enqueue_limit: int = None):
    """
    Read centroids from a file and enqueue them to SQS for processing.
    """
    with open(filename, 'r') as fh:
        for i, line in enumerate(tqdm(fh)):
            if enqueue_limit is not None and i >= enqueue_limit:
                break
            centroid_xyz = tuple(map(int, line.strip().split(',')))
            payload = {
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


def generate_cuboidwise_tasks_for_contactome(sqs_url: str, segmentation_channel: str, mip: list, enqueue_limit: int = None):
    # Create a file with each line being a cuboid start and radius
    seg_data = CloudVolume(segmentation_channel, mip=mip, cache=True)
    blocks = block_compute(
        x_start=0,
        x_stop=int(seg_data.shape[0]),
        y_start=0,
        y_stop=int(seg_data.shape[1]),
        z_start=0,
        z_stop=int(seg_data.shape[2]),
        block_size=(64, 64, 64),
    )
    print(f"Queueing {len(blocks)} blocks")
    # exit(1)
    for i, ((x_start, x_stop), (y_start, y_stop), (z_start, z_stop)) in enumerate(blocks):
        if enqueue_limit is not None and i >= enqueue_limit:
            break

        # enqueue a ContactomeEdgeTaskPayload
        payload: ContactomeEdgeTaskPayload = {
            "graph_id": "example_graph_id",
            "cuboid_start": (x_start, y_start, z_start),
            "cuboid_radius": (x_stop - x_start, y_stop - y_start, z_stop - z_start),
            "segmentation_channel": segmentation_channel,
            "mip": mip
        }
        sqs.send_message(
            QueueUrl=sqs_url,
            MessageBody=json.dumps(payload)
        )


def local_dequeue():
    import cloudome
    response = sqs.receive_message(
        QueueUrl="https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
        MaxNumberOfMessages=1
    )
    if "Messages" in response:
        for message in response["Messages"]:
            cloudome.process_queue_job({"Records": [message]}, None)
            sqs.delete_message(
                QueueUrl="https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
                ReceiptHandle=message["ReceiptHandle"]
            )


def initialize_resources():
    # TODO: Also provision SQS at some point...
    SynapseEdgeResultsModel.create_table(
        billing_mode="PAY_PER_REQUEST",
    )


def export_results_to_csv(graph_id: str, output_file: str):
    """
    Export results for a given graph_id to a CSV file.
    This function streams the results to handle large datasets efficiently.
    """
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['graph_id', 'synapse_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Stream results from SynapseEdgeResultsModel
        for result in SynapseEdgeResultsModel.query(graph_id):
            writer.writerow({'graph_id': result.graph_id, 'synapse_id': result.synapse_id})

        # Stream results from ContactEdgeResultsModel
        for result in ContactEdgeResultsModel.query(graph_id):
            writer.writerow({'graph_id': result.graph_id, 'synapse_id': result.synapse_id})


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cloudome Command Line Interface")
    subparsers = parser.add_subparsers(dest="namespace", required=True)

    # Namespace: synapses
    synapses_parser = subparsers.add_parser("synapses", help="Commands related to synapses")
    synapses_subparsers = synapses_parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate (synapses)
    synapses_subparsers.add_parser("generate", help="Generate centroids for synapse mask")

    # Subcommand: enqueue (synapses)
    enqueue_parser = synapses_subparsers.add_parser("enqueue", help="Enqueue centroids from file")
    enqueue_parser.add_argument("--enqueue-limit", type=int, default=None, help="Limit the number of centroids to enqueue")

    # Namespace: contactome
    contactome_parser = subparsers.add_parser("contactome", help="Commands related to contactome")
    contactome_subparsers = contactome_parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate (contactome)
    contactome_generate_parser = contactome_subparsers.add_parser("generate", help="Generate cuboidwise tasks for contactome")
    contactome_generate_parser.add_argument("--enqueue-limit", type=int, default=None, help="Limit the number of tasks to enqueue")

    # Namespace: export
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument("graph_id", type=str, help="Graph ID to export results for")
    export_parser.add_argument("output_file", type=str, help="Output CSV file path")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.namespace == "synapses":
        if args.command == "generate":
            get_centroids_for_syn_mask("s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses", mip=MIP)
        elif args.command == "enqueue":
            enqueue_centroids_from_file(
                sqs_url="https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
                graph_id="example_graph_id",
                filename="../centroids.csv",
                synapse_channel="s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses",
                segmentation_channel="s3://cvdb-bossdb-boss/smith2024/zebrafish/agglomeration_checkpoint_40000",
                mip=MIP,
                enqueue_limit=args.enqueue_limit
            )
    elif args.namespace == "contactome":
        if args.command == "generate":
            generate_cuboidwise_tasks_for_contactome(
                sqs_url="https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
                segmentation_channel="s3://cvdb-bossdb-boss/smith2024/zebrafish/agglomeration_checkpoint_40000",
                mip=MIP,
                enqueue_limit=args.enqueue_limit
            )
    elif args.namespace == "export":
        export_results_to_csv(args.graph_id, args.output_file)


if __name__ == "__main__":
    main()