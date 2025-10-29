import argparse
import cc3d
from intern.utils.parallel import block_compute
from cloudvolume import CloudVolume
from tqdm import tqdm
import numpy as np


from database import SynapseEdgeTaskPayload, TaskType


def export_synapse_mask_centroids_to_file(
    synapse_channel: str, output_file: str, mip: list | int
):
    # Use post synaptic densities as centroids. One synapse per PSD
    binary_syn_mask = (
        CloudVolume(synapse_channel, mip=mip, cache=True)[..., 0].squeeze() == 1
    )
    labels_out, N = cc3d.connected_components(binary_syn_mask, return_N=True)

    dust_threshold = 1

    stats = cc3d.statistics(labels_out)
    with open(output_file, "w") as fh:
        for i, syn_centroid in tqdm(enumerate(stats["centroids"])):
            if np.any(np.isnan(syn_centroid)):
                continue
            size = stats["voxel_counts"][i]
            if size > dust_threshold:
                fh.write(",".join(map(str, map(int, syn_centroid))) + "\n")


def generate_centroidwise_tasks(
    graph_id: str,
    filename: str,
    synapse_channel: str,
    segmentation_channel: str,
    mip: list | int,
    enqueue_limit: int | None = None,
):
    """
    Read centroids from a file and enqueue them to SQS for processing.
    """
    with open(filename, "r") as fh:
        for i, line in enumerate(tqdm(fh)):
            if enqueue_limit is not None and i >= enqueue_limit:
                break
            centroid_xyz: tuple[float, float, float] = tuple(
                map(float, line.strip().split(","))
            )
            payload: SynapseEdgeTaskPayload = {
                "graph_id": graph_id,
                "centroid_xyz": centroid_xyz,
                "synapse_channel": synapse_channel,
                "segmentation_channel": segmentation_channel,
                "mip": mip,
            }
            yield payload
