from typing import Literal, TypedDict

from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute

TaskType = Literal["connectome", "contactome", "volume", "supervoxel"]
MipType = list[float] | int

CentroidXYZ = tuple[float, float, float]


class SynapseEdgeTask(TypedDict):
    centroid_xyz: CentroidXYZ
    synapse_channel: str
    segmentation_channel: str
    mip: MipType


class SynapseEdgeTaskPayload(TypedDict):
    graph_id: str
    centroid_xyz: CentroidXYZ
    synapse_channel: str
    segmentation_channel: str
    mip: MipType


class ContactomeEdgeTaskPayload(TypedDict):
    graph_id: str
    task_type: TaskType
    cuboid_start: CentroidXYZ
    cuboid_radius: CentroidXYZ
    segmentation_channel: str
    mip: MipType


class VolumeTaskPayload(TypedDict):
    graph_id: str
    task_type: TaskType
    cuboid_start: CentroidXYZ
    cuboid_radius: CentroidXYZ
    segmentation_channel: str
    mip: MipType


class SupervoxelTaskPayload(TypedDict):
    graph_id: str
    task_type: TaskType  # "supervoxel"
    cuboid_start: CentroidXYZ  # chunk origin (x, y, z)
    cuboid_radius: CentroidXYZ  # chunk dimensions (x, y, z)
    chunk_index_xyz: tuple[int, int, int]  # (cx, cy, cz) for global ID packing
    n_chunks_xyz: tuple[int, int, int]  # total chunks (nx, ny, nz) for ID packer init
    segmentation_channel: str  # input segmentation
    output_channel: str  # destination for supervoxels
    raw_channel: str | None  # optional guidance for watershed
    mip: MipType
    target_voxels_per_sv: int  # ~25000
    min_voxels_per_sv: int  # ~2000
    halo: int  # ~8 for boundary handling
    edge_sigma: float  # ~1.5 for gradient
    min_local_bits: int  # allocate bits for local IDs


class SynapseEdgeResultsModel(Model):
    """
    A DynamoDB store for results
    """

    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True)  # "syn_x100_y20_z42_pre150_post60"


class ContactEdgeResultsModel(Model):
    """
    A DynamoDB store for results
    """

    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(
        range_key=True
    )  # "ctc_x100_y20_z42_pre150_post60_w100"


class VolumeCountResultsModel(Model):
    """
    A DynamoDB store for results
    """

    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True)  # "vol_x100_y20_z42_seg19934_v1263"


class SupervoxelResultsModel(Model):
    """
    A DynamoDB store for supervoxel chunk processing results
    """

    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(
        range_key=True
    )  # "sv_x0_y0_z0_n_chunks_248_num_sv_15_parent_counts_..."
