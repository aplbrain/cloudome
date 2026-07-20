import argparse
from typing import Any, cast

import cc3d
from cloudvolume import CloudVolume
from tqdm import tqdm
import numpy as np


from .database import SynapseEdgeTaskPayload


def export_synapse_mask_centroids_to_file(
    synapse_channel: str, output_file: str, mip: list | int
):
    # Use post synaptic densities as centroids. One synapse per PSD
    synapse_volume = CloudVolume(synapse_channel, mip=cast(Any, mip), cache=True)
    voxel_offset = np.array(
        tuple(getattr(synapse_volume, "voxel_offset"))[:3], dtype=np.float64
    )
    binary_syn_mask = (
        np.asarray(synapse_volume[..., 0]).squeeze() == 1  # type: ignore[index]
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
                adjusted = np.array(syn_centroid, dtype=np.float64) + voxel_offset
                adjusted_ints = tuple(int(value) for value in adjusted)
                fh.write(",".join(str(value) for value in adjusted_ints) + "\n")


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

            parts = [float(coord) for coord in line.strip().split(",")]
            if len(parts) != 3:
                raise ValueError(
                    "Expected centroid line to contain exactly three comma-separated values."
                )
            centroid_xyz: tuple[float, float, float] = (
                parts[0],
                parts[1],
                parts[2],
            )
            payload: SynapseEdgeTaskPayload = {
                "graph_id": graph_id,
                "centroid_xyz": centroid_xyz,
                "synapse_channel": synapse_channel,
                "segmentation_channel": segmentation_channel,
                "mip": mip,
            }
            yield payload
