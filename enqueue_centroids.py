import sys
from functools import partial
import json
import boto3
from tqdm.auto import tqdm
from cloudvolume import CloudVolume
import cc3d

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


def enqueue_centroids_from_file(sqs_url: str, graph_id: str, filename: str, synapse_channel: str, segmentation_channel: str, mip: list):
    """
    Read centroids from a file and enqueue them to SQS for processing.
    """
    with open(filename, 'r') as fh:
        for line in tqdm(fh):
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
            return



def initialize_resources():
    # TODO: Also provision SQS at some point...
    ResultsModel.create_table(
        billing_mode="PAY_PER_REQUEST",
    )

if __name__ == "__main__":
    cmd = sys.argv[-1]

    commands = {
        "init": partial(initialize_resources),
        "generate": partial(get_centroids_for_syn_mask, "s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses", mip=MIP),
        "enqueue": partial(
            enqueue_centroids_from_file,
            sqs_url="https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
            graph_id="example_graph_id",
            filename="../centroids.csv",
            synapse_channel="s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses",
            segmentation_channel="s3://cvdb-bossdb-boss/smith2024/zebrafish/agglomeration_checkpoint_40000",
            mip=MIP
        )
    }
    if cmd not in commands:
        raise ValueError("Command not recognized. Available commands are: " + ", ".join(commands.keys()))
    commands[cmd]()