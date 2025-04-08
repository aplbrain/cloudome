import os
import json
import numpy as np
from flask import Flask
from typing import TypedDict
from database import ResultsModel

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

def process_queue_job(event, context):
    payload: SynapseEdgeTaskPayload = json.loads(event['Records'][0]['body'])
    graph_id = payload.pop("graph_id")
    u, v = return_seg_edge(payload)
    # Save edge to dynamodb
    ResultsModel(
        graph_id=graph_id,
        synapse_id=f"pre{u}_post{v}_x{payload['centroid_xyz'][0]}_y{payload['centroid_xyz'][1]}_z{payload['centroid_xyz'][2]}"
    ).save()


