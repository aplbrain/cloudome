from cloudvolume import CloudVolume
from intern.utils.parallel import block_compute

from database import ContactomeEdgeTaskPayload, TaskType, VolumeTaskPayload


def generate_cuboidwise_tasks(
    graph_id: str,
    task_type: TaskType,
    segmentation_channel: str,
    mip: list | int,
    block_size: tuple = (64, 64, 64),
    z_start: int | None = None,
    z_end: int | None = None,
    enqueue_limit: int | None = None,
):
    # Create a file with each line being a cuboid start and radius
    seg_data = CloudVolume(
        segmentation_channel, mip=mip, cache=True, use_https=True, secrets=""
    )

    voxel_offset_raw = getattr(seg_data, "voxel_offset")
    voxel_offset = tuple(int(v) for v in tuple(voxel_offset_raw)[:3])
    shape_raw = getattr(seg_data, "shape")
    volume_shape = tuple(int(s) for s in tuple(shape_raw)[:3])

    x_start = voxel_offset[0]
    x_stop = voxel_offset[0] + volume_shape[0]
    y_start = voxel_offset[1]
    y_stop = voxel_offset[1] + volume_shape[1]

    z_start_voxel = 0 if z_start is None else max(0, min(volume_shape[2], z_start))
    if z_end is None:
        z_end_voxel = volume_shape[2]
    else:
        z_end_voxel = max(z_start_voxel, min(volume_shape[2], z_end))

    z_start = voxel_offset[2] + z_start_voxel
    z_stop = voxel_offset[2] + z_end_voxel

    blocks = block_compute(
        x_start=x_start,
        x_stop=x_stop,
        y_start=y_start,
        y_stop=y_stop,
        z_start=z_start,
        z_stop=z_stop,
        block_size=block_size,
    )

    if enqueue_limit:
        print(f"Queueing {min(len(blocks), enqueue_limit)} blocks")
    else:
        print(f"Queueing {len(blocks)} blocks")

    for i, ((x_start, x_stop), (y_start, y_stop), (z_start, z_stop)) in enumerate(
        blocks
    ):
        if enqueue_limit is not None and i >= enqueue_limit:
            break

        # enqueue a ContactomeEdgeTaskPayload
        payload: ContactomeEdgeTaskPayload | VolumeTaskPayload = {
            "graph_id": graph_id,
            "task_type": task_type,
            "cuboid_start": (x_start, y_start, z_start),
            "cuboid_radius": (x_stop - x_start, y_stop - y_start, z_stop - z_start),
            "segmentation_channel": segmentation_channel,
            "mip": mip,
        }  # type: ignore
        yield payload
