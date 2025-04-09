import os
import json
import numpy as np
from flask import Flask
from typing import TypedDict
from database import SynapseEdgeResultsModel, ContactEdgeResultsModel

os.environ['CLOUD_VOLUME_DIR'] = '/tmp/cloudvolume'
os.makedirs('/tmp/cloudvolume', exist_ok=True)
from cloudvolume import CloudVolume

SegmentID = int
CentroidXYZ = tuple[float, float, float]
app = Flask(__name__)

@app.route("/")
def _():
    return "Cloudome 2025-04-03"

class SynapseEdgeTask(TypedDict):
    centroid_xyz: CentroidXYZ
    synapse_channel: str
    segmentation_channel: str
    mip: list

class SynapseEdgeTaskPayload(TypedDict):
    graph_id: str
    centroid_xyz: CentroidXYZ
    synapse_channel: str
    segmentation_channel: str
    mip: list

class ContactomeEdgeTaskPayload(TypedDict):
    graph_id: str
    cuboid_start: CentroidXYZ
    cuboid_radus: CentroidXYZ
    segmentation_channel: str
    mip: list


RADIUS = 32
PRESYNAPTIC = 2
POSTSYNAPTIC = 1

def return_seg_edge(task: SynapseEdgeTask) -> tuple[SegmentID, SegmentID]:
    xyz_center = task['centroid_xyz']
    try:
        # Get the CloudVolume dimensions
        synapse_volume = CloudVolume(task['synapse_channel'], use_https=True, cache=False, secrets="", mip=task['mip'])
        segmentation_volume = CloudVolume(task['segmentation_channel'], use_https=True, cache=False, secrets="", mip=task['mip'])

        bounds = synapse_volume.shape
        x_min, x_max = max(0, xyz_center[0] - RADIUS), min(bounds[0], xyz_center[0] + RADIUS)
        y_min, y_max = max(0, xyz_center[1] - RADIUS), min(bounds[1], xyz_center[1] + RADIUS)
        z_min, z_max = max(0, xyz_center[2] - RADIUS), min(bounds[2], xyz_center[2] + RADIUS)

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")

        prepost_mask = synapse_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()
        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()

        # Count seg voxels per id in pre, get ID with most common count => pre_id
        pre_max_id = np.unique(seg_mask[prepost_mask == PRESYNAPTIC], return_counts=True)
        if pre_max_id[0].size == 0:
            raise ValueError("No presynaptic ID pixels found at {}.".format(xyz_center))
            pre_max_id = (-1,)
        else:
            pre_max_id = pre_max_id[0][np.argmax(pre_max_id[1])]

        # Count seg voxels per id in post, get ID with most common count => post_id
        post_max_id = np.unique(seg_mask[prepost_mask == POSTSYNAPTIC], return_counts=True)
        if post_max_id[0].size == 0:
            raise ValueError("No postsynaptic ID pixels found at {}.".format(xyz_center))
            post_max_id = (-1,)
        else:
            post_max_id = post_max_id[0][np.argmax(post_max_id[1])]

        return pre_max_id, post_max_id
    except Exception as e:
        print(f"[ERROR]\t{e}")
        return -1, -1

def count_contact_voxels(segmentation):
    segment_ids = np.unique(segmentation)
    contact_counts = {i: {j: 0 for j in segment_ids if j != i} for i in segment_ids}

    # Loop over all pairs of segment IDs, creating a mask for each segment and
    # counting the number of voxels that are in contact between the two masks.
    # "Contact" here is defined by shifting the mask in all cardinal directions
    # and taking the union of the shifted masks with the unshifted.
    #
    # In one dimension, that looks like this:
    #
    # i_mask = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    # j_mask = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    #
    # Right shift j-mask:
    # i_mask = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    # j_mask = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    # Union of the two is 0.
    #
    # Left shift j-mask:
    # i_mask = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    # j_mask = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    #                    ^
    # Union of the two is 1.
    # Thus, 0+1 = 1 contact voxel.
    for i in segment_ids:
        i_mask = segmentation == i
        # 0-pad the mask so that we can roll it in all directions
        i_mask = np.pad(i_mask, 1, mode="constant", constant_values=0)
        for j in segment_ids:
            if i != j:
                j_mask = segmentation == j
                j_mask = np.pad(j_mask, 1, mode="constant", constant_values=0)
                contact_counts[i][j] = np.sum(
                    i_mask & np.roll(j_mask, 1, axis=0)
                    | i_mask & np.roll(j_mask, -1, axis=0)
                    | i_mask & np.roll(j_mask, 1, axis=1)
                    | i_mask & np.roll(j_mask, -1, axis=1)
                    | i_mask & np.roll(j_mask, 1, axis=2)
                    | i_mask & np.roll(j_mask, -1, axis=2)
                )
    return contact_counts

def return_ctc_edges(task: ContactomeEdgeTaskPayload):
    xyz_start = task['cuboid_start']
    xyz_radius = task['cuboid_radus']
    try:
        # Get the CloudVolume dimensions
        segmentation_volume = CloudVolume(task['segmentation_channel'], use_https=True, cache=False, secrets="", mip=task['mip'])

        bounds = segmentation_volume.shape
        x_min, x_max = max(0, xyz_start[0] - xyz_radius[0]), min(bounds[0], xyz_start[0] + xyz_radius[0])
        y_min, y_max = max(0, xyz_start[1] - xyz_radius[1]), min(bounds[1], xyz_start[1] + xyz_radius[1])
        z_min, z_max = max(0, xyz_start[2] - xyz_radius[2]), min(bounds[2], xyz_start[2] + xyz_radius[2])

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")

        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()

        # Count seg voxels per id in pre, get ID with most common count => pre_id
        contact_counts = count_contact_voxels(seg_mask)
        edges = []
        for pre_id, post_counts in contact_counts.items():
            for post_id, count in post_counts.items():
                edges.append((pre_id, post_id, count))
        return edges
    except Exception as e:
        print(f"[ERROR]\t{e}")
        return []


def process_queue_job(event, context):
    payload = json.loads(event['Records'][0]['body'])
    if "cuboid_start" in payload:
        # Contactome edge
        payload = ContactomeEdgeTaskPayload(**payload)
        graph_id = payload.pop("graph_id")
        # get edges and weights:
        edges = return_ctc_edges(payload)
        for (pre, post, count) in edges:
            # Save edge to dynamodb
            ContactEdgeResultsModel(
                graph_id=graph_id,
                # XYZ goes first so that it can still serve as a useful key to retrieve
                # a specific centroid from the listing:
                synapse_id=f"ctc_x{payload['cuboid_start'][0]}_y{payload['cuboid_start'][1]}_z{payload['cuboid_start'][2]}_pre{pre}_post{post}_w{count}"
            ).save()

    else:
        # Synapse edge
        payload = SynapseEdgeTaskPayload(**payload)
        graph_id = payload.pop("graph_id")
        u, v = return_seg_edge(payload)
        # Save edge to dynamodb
        SynapseEdgeResultsModel(
            graph_id=graph_id,
            # XYZ goes first so that it can still serve as a useful key to retrieve
            # a specific centroid from the listing:
            synapse_id=f"syn_x{payload['centroid_xyz'][0]}_y{payload['centroid_xyz'][1]}_z{payload['centroid_xyz'][2]}_pre{u}_post{v}"
        ).save()


