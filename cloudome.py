import json
from flask import Flask
from typing import TypedDict

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


def return_seg_edge(task: SynapseEdgeTask) -> tuple[SegmentID, SegmentID]:
    # Pull a small volume ± 256px
    # Get the mask of pre/post
    # Count seg voxels per id in pre, get max => pre_id
    # Count seg voxels per id in post, get max => post_id
    # return (pre_id, post_id)
    return (1, 2)

def process_queue_job(event, context):
    payload = json.loads(event['Records'][0]['body'])
    graph_id = payload.pop("graph_id")
    u, v = return_seg_edge(payload)
    # Save edge to dynamodb
    db.save({
        u, v, graph_id
    })


