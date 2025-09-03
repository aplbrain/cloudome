import os
import json
import numpy as np
from flask import Flask
from database import (
    SynapseEdgeTask,
    SynapseEdgeTaskPayload,
    ContactomeEdgeTaskPayload,
    SynapseEdgeResultsModel,
    ContactEdgeResultsModel,
    VolumeCountResultsModel,
    VolumeTaskPayload,
)
import cc3d
import math

os.environ['CLOUD_VOLUME_DIR'] = '/tmp/cloudvolume'
os.makedirs('/tmp/cloudvolume', exist_ok=True)
from cloudvolume import CloudVolume

SegmentID = int

app = Flask(__name__)

@app.route("/")
def _():
    return "Cloudome 2025-04-03"


RADIUS = 10
PRESYNAPTIC = 2
POSTSYNAPTIC = 1

def return_seg_edge(task: SynapseEdgeTaskPayload) -> tuple[SegmentID, SegmentID]:
    xyz_center = task['centroid_xyz']
    try:
        # Get the CloudVolume dimensions
        synapse_volume = CloudVolume(task['synapse_channel'], use_https=True, cache=False, secrets="", mip=task['mip'])
        segmentation_volume = CloudVolume(task['segmentation_channel'], use_https=True, cache=False, secrets="", mip=task['mip'])

        # Calculate bounding box
        bounds = synapse_volume.shape
        x_min, x_max = max(0, xyz_center[0] - RADIUS), min(bounds[0], xyz_center[0] + RADIUS)
        y_min, y_max = max(0, xyz_center[1] - RADIUS), min(bounds[1], xyz_center[1] + RADIUS)
        z_min, z_max = max(0, xyz_center[2] - RADIUS), min(bounds[2], xyz_center[2] + RADIUS)

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")

        # Pull volumes inside bounding box for both synapse and segmentation paint
        prepost_mask = synapse_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()
        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()

        # Count seg voxels per id in pre, get ID with most common count => pre_id
        vals, counts = np.unique(seg_mask[prepost_mask == PRESYNAPTIC], return_counts=True)
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
        for i, syn_centroid in enumerate(stats['centroids']):
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
            raise ValueError("No postsynaptic ID pixels found at {}.".format(xyz_center))
        else:
            post_max_id = vals[0]
            # Throw out id zero and presyn id if they are in indices 0 and/or 1
            if (post_max_id == 0 or post_max_id == pre_max_id) and len(vals) > 1:
                post_max_id = vals[1]
                if (post_max_id == 0 or post_max_id == pre_max_id) and len(vals) > 2:
                    post_max_id = vals[2]
            # If no postsynaptic ID is found, add in contact voxels
            if (post_max_id == 0 or post_max_id == pre_max_id):
                label_mask_encoding = 1
                label_mask = (labels_out == label)
                masked_synapse_seg_vol = seg_mask 
                masked_synapse_seg_vol[label_mask] = label_mask_encoding
                contacts = cc3d.contacts(masked_synapse_seg_vol, connectivity=26)
                max_contact = 0
                max_contact_id = -1
                for contact in contacts:
                    if (label_mask_encoding in contact) and (pre_max_id not in contact) and (contacts[contact] > max_contact):
                        max_contact = contacts[contact]
                        max_contact_id = contact[1]
                post_max_id = max_contact_id

        return pre_max_id, post_max_id

    except Exception as e:
        print(f"[ERROR]\t{e}")
        return -1, -1

def count_contact_voxels(segmentation, resolution):
    contacts = cc3d.contacts(segmentation,
                             connectivity=6,
                             anisotropy=tuple(resolution), 
                             surface_area=True
    )
    return contacts

def count_volume_voxels(segmentation) -> dict[SegmentID, int]:
    """
    Count the number of voxels for each segment ID in the segmentation volume.
    """
    segment_ids = np.unique(segmentation)
    volume_counts = {i: 0 for i in segment_ids}

    for i in segment_ids:
        volume_counts[i] = np.sum(segmentation == i)

    return volume_counts

def return_ctc_edges(task: ContactomeEdgeTaskPayload):
    xyz_start = task['cuboid_start']
    print(xyz_start)
    xyz_radius = task['cuboid_radius']
    try:
        # Get the CloudVolume dimensions
        segmentation_volume = CloudVolume(task['segmentation_channel'], use_https=True, parallel=False, cache=False, secrets="", mip=task['mip'])

        bounds = segmentation_volume.shape
        # Add +1 to each leading edge coord so that contacts with adjacent cuboids are properly recorded
        x_min, x_max = max(0, xyz_start[0]), min(bounds[0], xyz_start[0] + xyz_radius[0] + 1)
        y_min, y_max = max(0, xyz_start[1]), min(bounds[1], xyz_start[1] + xyz_radius[1] + 1)
        z_min, z_max = max(0, xyz_start[2]), min(bounds[2], xyz_start[2] + xyz_radius[2] + 1)

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")

        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()

        # Count seg voxels per id in pre, get ID with most common count => pre_id
        contact_counts = count_contact_voxels(seg_mask, task['mip'])
        # edges = []
        # for pre_id, post_counts in contact_counts.items():
        #     if pre_id > 0:
        #         for post_id, count in post_counts.items():
        #             if count > 0 and pre_id != post_id and post_id > 0:
        #                 edges.append((pre_id, post_id, count))
        return contact_counts
    except Exception as e:
        print(f"[ERROR]\t[ctc] {e}")
        return []

def return_volume_counts(task: VolumeTaskPayload):
    xyz_start = task['cuboid_start']
    xyz_radius = task['cuboid_radius']
    try:
        # Get the CloudVolume dimensions
        segmentation_volume = CloudVolume(task['segmentation_channel'], use_https=True, parallel=False, cache=False, secrets="", mip=task['mip'])

        bounds = segmentation_volume.shape
        x_min, x_max = max(0, xyz_start[0]), min(bounds[0], xyz_start[0] + xyz_radius[0])
        y_min, y_max = max(0, xyz_start[1]), min(bounds[1], xyz_start[1] + xyz_radius[1])
        z_min, z_max = max(0, xyz_start[2]), min(bounds[2], xyz_start[2] + xyz_radius[2])

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValueError("Slicing range is invalid due to out-of-bounds coordinates.")

        seg_mask = segmentation_volume[x_min:x_max, y_min:y_max, z_min:z_max, 0].squeeze()

        volume_counts = count_volume_voxels(seg_mask)
        return volume_counts
    except Exception as e:
        print(f"[ERROR]\t[volume] {e}")
        return {}

def process_queue_job(event, context):
    # event_records = json.loads(event['Records'][0]['body'])
    for record in event['Records']:
        payload = json.loads(record['body'])
        # Uses "contactome" as the default task_type for back-compat.
        if "cuboid_start" in payload and payload.get("task_type", "contactome") == "contactome":
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
                    synapse_id=f"ctc_x{payload['cuboid_start'][0]}_y{payload['cuboid_start'][1]}_z{payload['cuboid_start'][2]}_pre{ids[0]}_post{ids[1]}_w{edges[ids]}"
                ).save()

        elif "cuboid_start" in payload and payload.get("task_type", "contactome") == "volume":
            # Volume task
            payload = VolumeTaskPayload(**payload)
            graph_id = payload.pop("graph_id")
            volume_counts = return_volume_counts(payload)
            for (seg_id, count) in volume_counts.items():
                VolumeCountResultsModel(
                    graph_id=graph_id,
                    synapse_id=f"vol_x{payload['cuboid_start'][0]}_y{payload['cuboid_start'][1]}_z{payload['cuboid_start'][2]}_seg{seg_id}_v{count}"
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
                synapse_id=f"syn_x{payload['centroid_xyz'][0]}_y{payload['centroid_xyz'][1]}_z{payload['centroid_xyz'][2]}_pre{u}_post{v}"
            ).save()


