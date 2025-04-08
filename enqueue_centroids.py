import sys
from functools import partial
import json
import boto3
from tqdm.auto import tqdm
from cloudvolume import CloudVolume
import cc3d
import argparse

from database import ResultsModel


sqs = boto3.client('sqs', region_name='us-east-1')

MIP = [72, 72, 84]


def get_centroids_for_syn_mask(synapse_channel: str, mip: list|int):
    binary_syn_mask = (CloudVolume(synapse_channel, mip=mip, cache=True)[..., 0].squeeze() > 0)
    labels_out, N = cc3d.connected_components(binary_syn_mask, return_N=True)
    stats = cc3d.statistics(labels_out)
    try:
        with open("../centroids.csv", 'w') as fh:
            for syn_centroid in stats['centroids']:
                fh.write(",".join(map(str, map(int, syn_centroid))) + "\n")
    except:
        import pdb; pdb.set_trace()


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
    ResultsModel.create_table(
        billing_mode="PAY_PER_REQUEST",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cloudome Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: init
    subparsers.add_parser("init", help="Initialize resources")

    # Subcommand: generate
    subparsers.add_parser("generate", help="Generate centroids for synapse mask")

    # Subcommand: enqueue
    enqueue_parser = subparsers.add_parser("enqueue", help="Enqueue centroids from file")
    enqueue_parser.add_argument("--enqueue-limit", type=int, default=None, help="Limit the number of centroids to enqueue")

    # Subcommand: local-dequeue
    subparsers.add_parser("local-dequeue", help="Dequeue a task locally and process it")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.command == "init":
        initialize_resources()
    elif args.command == "generate":
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
    elif args.command == "local-dequeue":
        local_dequeue()


if __name__ == "__main__":
    main()