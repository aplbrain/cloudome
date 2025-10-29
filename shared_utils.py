from intern.utils.parallel import block_compute
from cloudvolume import CloudVolume
from tqdm import tqdm


from database import TaskType


def generate_cuboidwise_tasks(
    graph_id: str,
    task_type: TaskType,
    segmentation_channel: str,
    mip: list | int,
    block_size: tuple = (64, 64, 64),
    z_start: int = None,
    z_end: int = None,
    enqueue_limit: int = None,
):
    # Create a file with each line being a cuboid start and radius
    seg_data = CloudVolume(segmentation_channel, mip=mip, cache=True)

    if z_end:
        z_end = z_end if z_end < int(seg_data.shape[2]) else int(seg_data.shape[2])
    else:
        z_end = int(seg_data.shape[2])

    blocks = block_compute(
        x_start=0,
        x_stop=int(seg_data.shape[0]),
        y_start=0,
        y_stop=int(seg_data.shape[1]),
        z_start=z_start or 0,
        z_stop=z_end,
        block_size=block_size,
    )

    # if enqueue_limit:
    #     print(f"Queueing {min(len(blocks), enqueue_limit)} blocks")
    # else:
    #     print(f"Queueing {len(blocks)} blocks")

    for i, ((x_start, x_stop), (y_start, y_stop), (z_start, z_stop)) in enumerate(
        blocks
    ):
        if enqueue_limit is not None and i >= enqueue_limit:
            break

        # enqueue a ContactomeEdgeTaskPayload
        payload = {
            "graph_id": graph_id,
            "task_type": task_type,
            "cuboid_start": (x_start, y_start, z_start),
            "cuboid_radius": (x_stop - x_start, y_stop - y_start, z_stop - z_start),
            "segmentation_channel": segmentation_channel,
            "mip": mip,
        }  # ContactomeEdgeTaskPayload | VolumeTaskPayload
        yield payload
