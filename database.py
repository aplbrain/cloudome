from typing import TypedDict

from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute

CentroidXYZ = tuple[float, float, float]

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
    cuboid_radius: CentroidXYZ
    segmentation_channel: str
    mip: list

class SynapseEdgeResultsModel(Model):
    """
    A DynamoDB store for results
    """
    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True) # "syn_x100_y20_z42_pre150_post60"

class ContactEdgeResultsModel(Model):
    """
    A DynamoDB store for results
    """
    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True) # "ctc_x100_y20_z42_pre150_post60_w100"