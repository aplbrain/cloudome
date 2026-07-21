import os
import json
from typing import Any, cast

import numpy as np
from flask import Flask
from database import (
    SynapseEdgeTaskPayload,
    ContactomeEdgeTaskPayload,
    SynapseEdgeResultsModel,
    ContactEdgeResultsModel,
    VolumeCountResultsModel,
    VolumeTaskPayload,
    SupervoxelTaskPayload,
    SupervoxelResultsModel,
)
import cc3d
import math
from collections import Counter

from cloudvolume import CloudVolume
from shared_supervoxel_utils import (
    process_chunk,
    GlobalIDPacker,
)

os.environ["CLOUD_VOLUME_DIR"] = "/tmp/cloudvolume"
os.makedirs("/tmp/cloudvolume", exist_ok=True)

SegmentID = int
ContactPair = tuple[SegmentID, SegmentID]


def _vec3_from_any(value: Any) -> tuple[int, int, int]:
    """Best-effort conversion of a CloudVolume Vec/Bbox component to integer xyz tuple."""
    try:
        iterable = list(value)  # type: ignore[arg-type]
    except TypeError:
        iterable = [
            getattr(value, "x"),
            getattr(value, "y"),
            getattr(value, "z"),
        ]
    if len(iterable) < 3:
        raise ValueError("Expected a three-component vector.")
    return (int(iterable[0]), int(iterable[1]), int(iterable[2]))


def _bbox_min_max(bounds: Any) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    minpt = getattr(bounds, "minpt")
    maxpt = getattr(bounds, "maxpt")
    return _vec3_from_any(minpt), _vec3_from_any(maxpt)


app = Flask(__name__)


@app.route("/")
def _():
    return "Cloudome 2025-04-03"


RADIUS = 10
PRESYNAPTIC = 2
POSTSYNAPTIC = 1


def return_seg_edge(task: SynapseEdgeTaskPayload) -> tuple[SegmentID, SegmentID]:
    xyz_center = task["centroid_xyz"]
    try:
        # Get the CloudVolume dimensions
        synapse_volume = CloudVolume(
            task["synapse_channel"],
            use_https=True,
            cache=False,
            mip=cast(int, task["mip"]),
            fill_missing=True,
        )
        segmentation_volume = CloudVolume(
            task["segmentation_channel"],
            use_https=True,
            cache=False,
            mip=cast(int, task["mip"]),
            fill_missing=True,
        )

        # Calculate bounding box respecting voxel offsets
        syn_min, syn_max = _bbox_min_max(getattr(synapse_volume, "bounds"))
        seg_min, seg_max = _bbox_min_max(getattr(segmentation_volume, "bounds"))

        x_min = max(syn_min[0], seg_min[0], int(xyz_center[0] - RADIUS))
        x_max = min(syn_max[0], seg_max[0], int(xyz_center[0] + RADIUS))
        y_min = max(syn_min[1], seg_min[1], int(xyz_center[1] - RADIUS))
        y_max = min(syn_max[1], seg_max[1], int(xyz_center[1] + RADIUS))
        z_min = max(syn_min[2], seg_min[2], int(xyz_center[2] - RADIUS))
        z_max = min(syn_max[2], seg_max[2], int(xyz_center[2] + RADIUS))

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError(
                "Slicing range is invalid due to out-of-bounds coordinates."
            )

        # Pull volumes inside bounding box for both synapse and segmentation paint
        prepost_mask = np.asarray(
            cast(Any, synapse_volume)[x_min:x_max, y_min:y_max, z_min:z_max, 0]
        ).squeeze()
        seg_mask = np.asarray(
            cast(Any, segmentation_volume)[x_min:x_max, y_min:y_max, z_min:z_max, 0]
        ).squeeze()

        # Count seg voxels per id in pre, get ID with most common count => pre_id
        vals, counts = np.unique(
            seg_mask[prepost_mask == PRESYNAPTIC], return_counts=True
        )
        unique_counts = zip(counts, vals)
        unique_counts = sorted(unique_counts, reverse=True)
        counts, vals = zip(*unique_counts)
        # Error handling for 0 case
        if len(vals) == 0:
            raise ValueError("No presynaptic ID pixels found at {}.".format(xyz_center))
        else:
            pre_max_id = vals[0]
            if pre_max_id == 0 and len(vals) > 1:
                pre_max_id = vals[1]
            elif pre_max_id == 0:
                raise ValueError("The only presynaptic ID returned is 0.")

        # Subsample PSD voxels to only the one we care about
        labels_out, N = cc3d.connected_components(prepost_mask, return_N=True)
        stats = cc3d.statistics(labels_out)
        distance = math.inf
        label = -1
        for i, syn_centroid in enumerate(stats["centroids"]):
            temp_distance = math.dist(syn_centroid, [RADIUS, RADIUS, RADIUS])
            if temp_distance < distance:
                distance = temp_distance
                label = i
        if label == -1:
            raise ValueError("No synapse centroids found in given subvolume.")

        # Count seg voxels per id in post, get ID with most common count => post_id
        vals, counts = np.unique(seg_mask[labels_out == label], return_counts=True)
        unique_counts = zip(counts, vals)
        unique_counts = sorted(unique_counts, reverse=True)
        counts, vals = zip(*unique_counts)
        # Error handling for 0 case
        if len(vals) == 0:
            raise ValueError(
                "No postsynaptic ID pixels found at {}.".format(xyz_center)
            )
        else:
            post_max_id = vals[0]
            # Throw out id zero and presyn id if they are in indices 0 and/or 1
            if (post_max_id == 0 or post_max_id == pre_max_id) and len(vals) > 1:
                post_max_id = vals[1]
                if (post_max_id == 0 or post_max_id == pre_max_id) and len(vals) > 2:
                    post_max_id = vals[2]
            # If no postsynaptic ID is found, add in contact voxels
            if post_max_id == 0 or post_max_id == pre_max_id:
                label_mask_encoding = 1
                label_mask = labels_out == label
                masked_synapse_seg_vol = seg_mask
                masked_synapse_seg_vol[label_mask] = label_mask_encoding
                contacts = cc3d.contacts(masked_synapse_seg_vol, connectivity=26)
                max_contact = 0
                max_contact_id = -1
                for contact in contacts:
                    if (
                        (label_mask_encoding in contact)
                        and (pre_max_id not in contact)
                        and (contacts[contact] > max_contact)
                    ):
                        max_contact = contacts[contact]
                        max_contact_id = contact[1]
                post_max_id = max_contact_id

        return pre_max_id, post_max_id

    except Exception as e:
        print(f"[ERROR]\t{e}")
        return -1, -1


def count_contact_voxels(segmentation, resolution) -> dict[ContactPair, int]:
    contacts = cc3d.contacts(
        segmentation, connectivity=6, anisotropy=tuple(resolution), surface_area=True
    )
    return cast(dict[ContactPair, int], contacts)


def remove_contact_overlap(
    contact_counts: dict[ContactPair, int],
    counts_to_remove: list[dict[ContactPair, int]],
):
    final_contacts = contact_counts
    for count_to_remove in counts_to_remove:
        final_contacts = dict(Counter(final_contacts) - Counter(count_to_remove))
    return final_contacts


def count_volume_voxels(segmentation) -> dict[SegmentID, int]:
    """
    Count the number of voxels for each segment ID in the segmentation volume.
    """
    segment_ids = np.unique(segmentation)
    volume_counts = {i: 0 for i in segment_ids}

    for i in segment_ids:
        volume_counts[i] = np.sum(segmentation == i)

    return volume_counts


def return_ctc_edges(task: ContactomeEdgeTaskPayload) -> dict[ContactPair, int]:
    xyz_start = task["cuboid_start"]
    xyz_radius = task["cuboid_radius"]
    try:
        # Get the CloudVolume dimensions
        segmentation_volume = CloudVolume(
            task["segmentation_channel"],
            use_https=True,
            parallel=False,
            cache=False,
            secrets="",
            mip=cast(Any, task["mip"]),
            fill_missing=True,
        )

        bounds_min, bounds_max = _bbox_min_max(getattr(segmentation_volume, "bounds"))

        # Add +1 to each leading edge coord so that contacts with adjacent cuboids are properly recorded
        x_min = max(bounds_min[0], int(xyz_start[0]))
        x_requested_max = int(xyz_start[0] + xyz_radius[0] + 1)
        x_max = min(bounds_max[0], x_requested_max)

        y_min = max(bounds_min[1], int(xyz_start[1]))
        y_requested_max = int(xyz_start[1] + xyz_radius[1] + 1)
        y_max = min(bounds_max[1], y_requested_max)

        z_min = max(bounds_min[2], int(xyz_start[2]))
        z_requested_max = int(xyz_start[2] + xyz_radius[2] + 1)
        z_max = min(bounds_max[2], z_requested_max)
        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError(
                "Slicing range is invalid due to out-of-bounds coordinates."
            )
        seg_mask = np.asarray(
            cast(Any, segmentation_volume)[x_min:x_max, y_min:y_max, z_min:z_max, 0]
        ).squeeze()

        # Generate contacts for whole volume
        resolution = _vec3_from_any(getattr(segmentation_volume, "resolution"))
        initial_contact_counts = count_contact_voxels(seg_mask, resolution)

        # Remove doubly-counted contacts at the edges
        o_x = count_contact_voxels(seg_mask[-1:, :, :], resolution)
        o_y = count_contact_voxels(seg_mask[:, -1:, :], resolution)
        o_z = count_contact_voxels(seg_mask[:, :, -1:], resolution)
        final_contact_counts = remove_contact_overlap(
            initial_contact_counts, [o_x, o_y, o_z]
        )

        return final_contact_counts
    except Exception as e:
        # print(f"[ERROR]\t[ctc] {e}")
        # Verbose:
        print(f"[ERROR]\t[ctc]. Exception traceback: {e}\n{type(e)}\t{e.args}")

        return {}


def return_volume_counts(task: VolumeTaskPayload):
    xyz_start = task["cuboid_start"]
    xyz_radius = task["cuboid_radius"]
    try:
        # Get the CloudVolume dimensions
        segmentation_volume = CloudVolume(
            task["segmentation_channel"],
            use_https=True,
            parallel=False,
            cache=False,
            secrets="",
            mip=cast(Any, task["mip"]),
            fill_missing=True,
        )

        bounds_min, bounds_max = _bbox_min_max(getattr(segmentation_volume, "bounds"))
        x_min = max(bounds_min[0], int(xyz_start[0]))
        x_max = min(bounds_max[0], int(xyz_start[0] + xyz_radius[0]))
        y_min = max(bounds_min[1], int(xyz_start[1]))
        y_max = min(bounds_max[1], int(xyz_start[1] + xyz_radius[1]))
        z_min = max(bounds_min[2], int(xyz_start[2]))
        z_max = min(bounds_max[2], int(xyz_start[2] + xyz_radius[2]))

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError(
                "Slicing range is invalid due to out-of-bounds coordinates."
            )

        seg_mask = np.asarray(
            cast(Any, segmentation_volume)[x_min:x_max, y_min:y_max, z_min:z_max, 0]
        ).squeeze()

        volume_counts = count_volume_voxels(seg_mask)
        return volume_counts
    except Exception as e:
        print(f"[ERROR]\t[volume] {e}")
        return {}


def return_supervoxel_results(task: SupervoxelTaskPayload) -> dict[str, Any]:
    """
    Process a supervoxel chunk task.
    Reads input segmentation + raw channel, processes the single chunk,
    writes output supervoxels to output_channel, and returns metadata.
    """
    # Load input segmentation with halo
    seg_cv = CloudVolume(
        task["segmentation_channel"],
        use_https=True,
        cache=False,
        secrets="",
        mip=cast(Any, task["mip"]),
        fill_missing=True,
    )
    raw_cv = None
    if task["raw_channel"]:
        raw_cv = CloudVolume(
            task["raw_channel"],
            use_https=True,
            cache=False,
            secrets="",
            mip=cast(Any, task["mip"]),
            fill_missing=False,
        )
    elif float(task["edge_sigma"]) > 0:
        raise ValueError(
            "raw_channel is required for membrane-aware supervoxelization when edge_sigma > 0."
        )
    # Use the dataset's chunk grid for aligned writes; task generation now matches this chunk size.
    bounds_min, bounds_max = _bbox_min_max(getattr(seg_cv, "bounds"))
    voxel_offset = _vec3_from_any(getattr(seg_cv, "voxel_offset"))
    chunk_size = _vec3_from_any(getattr(seg_cv, "chunk_size"))

    def _align_to_chunk(start: int, offset: int, chunk: int) -> int:
        return offset + ((start - offset) // chunk) * chunk

    x_start, y_start, z_start = task["cuboid_start"]
    halo = int(task["halo"])

    # Align to chunk grid (task chunk size is enforced to match dataset chunk size)
    ax = int(_align_to_chunk(int(x_start), voxel_offset[0], chunk_size[0]))
    ay = int(_align_to_chunk(int(y_start), voxel_offset[1], chunk_size[1]))
    az = int(_align_to_chunk(int(z_start), voxel_offset[2], chunk_size[2]))

    arx = int(task["cuboid_radius"][0])
    ary = int(task["cuboid_radius"][1])
    arz = int(task["cuboid_radius"][2])

    # Compute read bounds including halo, clamp to dataset bounds
    x_read_start = max(bounds_min[0], ax - halo)
    y_read_start = max(bounds_min[1], ay - halo)
    z_read_start = max(bounds_min[2], az - halo)
    x_read_stop = min(bounds_max[0], ax + arx + halo)
    y_read_stop = min(bounds_max[1], ay + ary + halo)
    z_read_stop = min(bounds_max[2], az + arz + halo)

    seg_data = np.squeeze(
        np.asarray(seg_cv[x_read_start:x_read_stop, y_read_start:y_read_stop, z_read_start:z_read_stop])
    )
    raw_data = None
    if raw_cv is not None:
        raw_data = np.squeeze(
            np.asarray(
                raw_cv[
                    x_read_start:x_read_stop,
                    y_read_start:y_read_stop,
                    z_read_start:z_read_stop,
                ]
            )
        )

    # Arrays are now in XYZ order (shape is X, Y, Z)
    # Prepare write-region chunks using true local offsets. This avoids dropping
    # voxels when halo reads are clipped at dataset boundaries.
    write_slice_x = slice(ax - x_read_start, ax - x_read_start + arx)
    write_slice_y = slice(ay - y_read_start, ay - y_read_start + ary)
    write_slice_z = slice(az - z_read_start, az - z_read_start + arz)

    seg_chunk = seg_data[write_slice_x, write_slice_y, write_slice_z]
    raw_chunk = (
        raw_data[write_slice_x, write_slice_y, write_slice_z]
        if raw_data is not None
        else None
    )

    id_packer = GlobalIDPacker(task["n_chunks_xyz"], min_local_bits=task["min_local_bits"])

    sv_chunk, metadata = process_chunk(
        seg_chunk,
        raw_chunk,
        task["chunk_index_xyz"],
        id_packer,
        target_voxels_per_sv=task["target_voxels_per_sv"],
        min_voxels_per_sv=task["min_voxels_per_sv"],
        edge_sigma=task["edge_sigma"],
        enable_force_dicing=False,
    )

    # Write supervoxels to output channel (write region only, no halo)
    output_cv = CloudVolume(
        task["output_channel"],
        cache=False,
        mip=cast(Any, task["mip"]),
    )
    
    output_cv[ax:ax + arx, ay:ay + ary, az:az + arz] = sv_chunk
    
    # Aggregate metadata (chunk index is already provided in the task)
    return {
        "num_sv": metadata["num_sv"],
        "parent_sv_ids": metadata["parent_sv_ids"],
        "sv_sizes": metadata["sv_sizes"],
        "chunk_index": tuple(task["chunk_index_xyz"]),
    }
    # except Exception as e:
    #     print(f"[ERROR]\t[supervoxel] {e}")
    #     return {}


def process_queue_job(event, context):
    # event_records = json.loads(event['Records'][0]['body'])
    for record in event["Records"]:
        payload = json.loads(record["body"])
        # Uses "contactome" as the default task_type for back-compat.
        if (
            "cuboid_start" in payload
            and payload.get("task_type", "contactome") == "supervoxel"
        ):
            # Supervoxel task
            payload = SupervoxelTaskPayload(**payload)
            graph_id = payload.pop("graph_id")
            chunk_index = payload["chunk_index_xyz"]
            results = return_supervoxel_results(payload)
            # Save results to dynamodb
            if results:
                # Encode parent->sv ids mapping as pid:gid1|gid2,... per parent, comma-separated parents
                parent_map = results.get("parent_sv_ids", {})
                parent_counts_str = ",".join(
                    f"{pid}:{'|'.join(str(int(g)) for g in gids)}" for pid, gids in parent_map.items()
                )
                SupervoxelResultsModel(
                    graph_id=graph_id,
                    synapse_id=f"sv_x{chunk_index[0]}_y{chunk_index[1]}_z{chunk_index[2]}_n{results['num_sv']}_p{parent_counts_str}",
                ).save()
        elif (
            "cuboid_start" in payload
            and payload.get("task_type", "contactome") == "contactome"
        ):
            # Contactome edge
            payload = ContactomeEdgeTaskPayload(**payload)
            graph_id = payload.pop("graph_id")
            # get edges and weights:
            edges = return_ctc_edges(payload)
            for ids in edges:
                # Save edge to dynamodb
                ContactEdgeResultsModel(
                    graph_id=graph_id,
                    # XYZ goes first so that it can still serve as a useful key to retrieve
                    # a specific centroid from the listing:
                    synapse_id=f"ctc_x{payload['cuboid_start'][0]}_y{payload['cuboid_start'][1]}_z{payload['cuboid_start'][2]}_pre{ids[0]}_post{ids[1]}_w{edges[ids]}",
                ).save()

        elif (
            "cuboid_start" in payload
            and payload.get("task_type", "contactome") == "volume"
        ):
            # Volume task
            payload = VolumeTaskPayload(**payload)
            graph_id = payload.pop("graph_id")
            volume_counts = return_volume_counts(payload)
            for seg_id, count in volume_counts.items():
                VolumeCountResultsModel(
                    graph_id=graph_id,
                    synapse_id=f"vol_x{payload['cuboid_start'][0]}_y{payload['cuboid_start'][1]}_z{payload['cuboid_start'][2]}_seg{seg_id}_v{count}",
                ).save()
        else:
            # Synapse edge
            payload = SynapseEdgeTaskPayload(**payload)
            graph_id = payload.pop("graph_id")
            u, v = return_seg_edge(payload)
            # # should probably just NOT insert -1's and 0's at all... too much clutter
            # if u in [0, -1] or v in [0, -1]:
            #     return
            # Save edge to dynamodb
            SynapseEdgeResultsModel(
                graph_id=graph_id,
                # XYZ goes first so that it can still serve as a useful key to retrieve
                # a specific centroid from the listing:
                synapse_id=f"syn_x{payload['centroid_xyz'][0]}_y{payload['centroid_xyz'][1]}_z{payload['centroid_xyz'][2]}_pre{u}_post{v}",
            ).save()
