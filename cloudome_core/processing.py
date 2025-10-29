from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, cast

import cc3d
import numpy as np
from cloudvolume import CloudVolume

from database import (  # type: ignore[import-not-found]
    ContactomeEdgeTaskPayload,
    SynapseEdgeTaskPayload,
    VolumeTaskPayload,
)

RADIUS = 10
PRESYNAPTIC = 2
POSTSYNAPTIC = 1


def _load_volume(path: str, mip: list[int] | int, *, parallel: bool = False) -> Any:
    """Helper to construct CloudVolume objects while appeasing static type checking."""

    kwargs: dict[str, Any] = {
        "use_https": True,
        "cache": False,
        "secrets": "",
        "mip": mip,
    }
    if parallel:
        kwargs["parallel"] = True
    return cast(Any, CloudVolume(path, **kwargs))


def return_seg_edge(task: SynapseEdgeTaskPayload) -> tuple[int, int]:
    xyz_center = task["centroid_xyz"]
    try:
        synapse_volume = _load_volume(task["synapse_channel"], task["mip"])
        segmentation_volume = _load_volume(task["segmentation_channel"], task["mip"])

        bounds = synapse_volume.shape
        x_min, x_max = max(0, xyz_center[0] - RADIUS), min(bounds[0], xyz_center[0] + RADIUS)
        y_min, y_max = max(0, xyz_center[1] - RADIUS), min(bounds[1], xyz_center[1] + RADIUS)
        z_min, z_max = max(0, xyz_center[2] - RADIUS), min(bounds[2], xyz_center[2] + RADIUS)

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")

        prepost_mask = synapse_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()
        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()

        values, counts = np.unique(seg_mask[prepost_mask == PRESYNAPTIC], return_counts=True)
        counts_with_ids = sorted(zip(counts, values), reverse=True)
        if not counts_with_ids:
            raise ValueError(f"No presynaptic ID pixels found at {xyz_center}.")
        pre_max_id = counts_with_ids[0][1]
        if pre_max_id == 0 and len(counts_with_ids) > 1:
            pre_max_id = counts_with_ids[1][1]
        elif pre_max_id == 0:
            raise ValueError("The only presynaptic ID returned is 0.")

        labels_out, _ = cc3d.connected_components(prepost_mask, return_N=True)
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

        values, counts = np.unique(seg_mask[labels_out == label], return_counts=True)
        counts_with_ids = sorted(zip(counts, values), reverse=True)
        if not counts_with_ids:
            raise ValueError(f"No postsynaptic ID pixels found at {xyz_center}.")
        post_max_id = counts_with_ids[0][1]
        if (post_max_id == 0 or post_max_id == pre_max_id) and len(counts_with_ids) > 1:
            post_max_id = counts_with_ids[1][1]
            if (post_max_id == 0 or post_max_id == pre_max_id) and len(counts_with_ids) > 2:
                post_max_id = counts_with_ids[2][1]
        if post_max_id == 0 or post_max_id == pre_max_id:
            label_mask_encoding = 1
            label_mask = labels_out == label
            masked_synapse_seg_vol = seg_mask.copy()
            masked_synapse_seg_vol[label_mask] = label_mask_encoding
            contacts = cc3d.contacts(masked_synapse_seg_vol, connectivity=26)
            max_contact = 0
            max_contact_id = -1
            for contact in contacts:
                if (
                    label_mask_encoding in contact
                    and pre_max_id not in contact
                    and contacts[contact] > max_contact
                ):
                    max_contact = contacts[contact]
                    max_contact_id = contact[1]
            post_max_id = max_contact_id

        return int(pre_max_id), int(post_max_id)

    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[ERROR]\t{exc}")
        return -1, -1


def count_contact_voxels(segmentation: np.ndarray, resolution: Iterable[float]):
    return cc3d.contacts(segmentation, connectivity=6, anisotropy=tuple(resolution), surface_area=True)


def remove_contact_overlap(
    contact_counts: dict[tuple[int, int], int],
    counts_to_remove: list[dict[tuple[int, int], int]],
):
    final_contacts = contact_counts
    for count_to_remove in counts_to_remove:
        final_contacts = dict(Counter(final_contacts) - Counter(count_to_remove))
    return final_contacts


def count_volume_voxels(segmentation: np.ndarray) -> dict[int, int]:
    segment_ids = np.unique(segmentation)
    volume_counts = {int(i): int(np.sum(segmentation == i)) for i in segment_ids}
    return volume_counts


def return_ctc_edges(task: ContactomeEdgeTaskPayload):
    xyz_start = task["cuboid_start"]
    try:
        segmentation_volume = _load_volume(task["segmentation_channel"], task["mip"], parallel=False)

        bounds = segmentation_volume.shape
        x_min, x_max = max(0, xyz_start[0]), min(bounds[0], xyz_start[0] + task["cuboid_radius"][0] + 1)
        y_min, y_max = max(0, xyz_start[1]), min(bounds[1], xyz_start[1] + task["cuboid_radius"][1] + 1)
        z_min, z_max = max(0, xyz_start[2]), min(bounds[2], xyz_start[2] + task["cuboid_radius"][2] + 1)
        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")
        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()

        initial_contact_counts = count_contact_voxels(seg_mask, segmentation_volume.resolution)

        o_x = count_contact_voxels(seg_mask[-1:, :, :], segmentation_volume.resolution)
        o_y = count_contact_voxels(seg_mask[:, -1:, :], segmentation_volume.resolution)
        o_z = count_contact_voxels(seg_mask[:, :, -1:], segmentation_volume.resolution)
        final_contact_counts = remove_contact_overlap(initial_contact_counts, [o_x, o_y, o_z])

        return final_contact_counts
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[ERROR]\t[ctc] {exc}")
        return {}


def return_volume_counts(task: VolumeTaskPayload):
    xyz_start = task["cuboid_start"]
    try:
        segmentation_volume = _load_volume(task["segmentation_channel"], task["mip"], parallel=False)

        bounds = segmentation_volume.shape
        x_min, x_max = max(0, xyz_start[0]), min(bounds[0], xyz_start[0] + task["cuboid_radius"][0])
        y_min, y_max = max(0, xyz_start[1]), min(bounds[1], xyz_start[1] + task["cuboid_radius"][1])
        z_min, z_max = max(0, xyz_start[2]), min(bounds[2], xyz_start[2] + task["cuboid_radius"][2])

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")

        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()
        volume_counts = count_volume_voxels(seg_mask)
        return volume_counts
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[ERROR]\t[volume] {exc}")
        return {}
