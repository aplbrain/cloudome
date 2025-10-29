from typing import Literal, TypedDict

from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute

TaskType = Literal["connectome", "contactome", "volume"]

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
    task_type: TaskType
    cuboid_start: CentroidXYZ
    cuboid_radius: CentroidXYZ
    segmentation_channel: str
    mip: list

class VolumeTaskPayload(TypedDict):
    graph_id: str
    task_type: TaskType
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

class VolumeCountResultsModel(Model):
    """
    A DynamoDB store for results
    """
    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True) # "vol_x100_y20_z42_seg19934_v1263"