"""
Supervoxel generation utilities.

Chunks are processed independently with halo regions for boundary affinities.
Global IDs are packed per-chunk to ensure no collisions across the volume.
"""

import math
from typing import Any, cast, Iterator, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed, relabel_sequential
from skimage.feature import peak_local_max
from cloudvolume import CloudVolume
from tqdm import tqdm

from .database import SupervoxelTaskPayload

SegmentID = int
ContactPair = tuple[SegmentID, SegmentID]


def bits_needed(n: int) -> int:
    """Minimum bits to represent values in [0, n-1]."""
    return 1 if n <= 1 else int(math.ceil(math.log2(n)))


class GlobalIDPacker:
    """
    Pack (cx, cy, cz, local_id) -> uint64 with no collisions; also decode back.
    cx, cy, cz are chunk indices (0-based).
    local_id is the 1..K index of the supervoxel inside that chunk.
    """

    def __init__(self, n_chunks_xyz: Tuple[int, int, int], min_local_bits: int = 20):
        nx, ny, nz = n_chunks_xyz
        bx = bits_needed(nx)
        by = bits_needed(ny)
        bz = bits_needed(nz)
        used = bx + by + bz
        remaining = 64 - used
        if remaining <= 0:
            raise ValueError(f"Not enough bits: {used} used by chunk indices.")
        self.bx, self.by, self.bz = bx, by, bz
        self.bl = max(1, min(min_local_bits, remaining))  # allocate at least 1 bit
        # If you want to force more bits for local IDs, ensure bx+by+bz+bl <= 64

        # Bit layout (most->least significant): [cx | cy | cz | local]
        self.shy = self.bz + self.bl
        self.shx = self.by + self.shy
        self.max_local = (1 << self.bl) - 1

    def encode(self, cx: int, cy: int, cz: int, local_id: int) -> np.uint64:
        if local_id < 1 or local_id > self.max_local:
            raise ValueError(
                f"local_id {local_id} exceeds allocated {self.bl} bits (max {self.max_local})."
            )
        return np.uint64(
            (cx << (self.by + self.shy))
            | (cy << self.shy)
            | (cz << self.bl)
            | local_id
        )

    def decode(self, gid: np.uint64) -> Tuple[int, int, int, int]:
        gid = int(gid)
        local_id = gid & ((1 << self.bl) - 1)
        cz = (gid >> self.bl) & ((1 << self.bz) - 1)
        cy = (gid >> self.shy) & ((1 << self.by) - 1)
        cx = (gid >> (self.by + self.shy)) & ((1 << self.bx) - 1)
        return cx, cy, cz, local_id


def chunk_grid_for_shape(
    shape_xyz: Tuple[int, int, int], chunk_xyz: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """Returns (n_chunks_x, n_chunks_y, n_chunks_z) given array shape in XYZ and chunk in XYZ."""
    X, Y, Z = shape_xyz
    cx, cy, cz = chunk_xyz
    nx = math.ceil(X / cx)
    ny = math.ceil(Y / cy)
    nz = math.ceil(Z / cz)
    return (nx, ny, nz)


def iter_chunk_bounds(
    shape_xyz: Tuple[int, int, int],
    chunk_xyz: Tuple[int, int, int],
    halo: int = 0,
) -> Iterator[Tuple[Tuple[int, int, int], Tuple[slice, slice, slice], Tuple[slice, slice, slice]]]:
    """
    Yield per-chunk (chunk_index_xyz, write_slices_xyz, read_slices_xyz).
    read_slices includes halo; write_slices excludes halo (hard boundary).
    All in XYZ order.
    """
    X, Y, Z = shape_xyz
    cx, cy, cz = chunk_xyz

    for cx_i, x0 in enumerate(range(0, X, cx)):
        for cy_i, y0 in enumerate(range(0, Y, cy)):
            for cz_i, z0 in enumerate(range(0, Z, cz)):
                x1 = min(x0 + cx, X)
                y1 = min(y0 + cy, Y)
                z1 = min(z0 + cz, Z)

                # write (no halo)
                wslc = (slice(x0, x1), slice(y0, y1), slice(z0, z1))

                # read (with halo, clipped to volume)
                rx0 = max(0, x0 - halo)
                ry0 = max(0, y0 - halo)
                rz0 = max(0, z0 - halo)
                rx1 = min(X, x1 + halo)
                ry1 = min(Y, y1 + halo)
                rz1 = min(Z, z1 + halo)
                rslc = (slice(rx0, rx1), slice(ry0, ry1), slice(rz0, rz1))

                yield (cx_i, cy_i, cz_i), wslc, rslc


def merge_small_regions(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    """
    Merge labels (< min_voxels) into the neighboring label with the largest shared boundary.
    labels: int array with 0=background, 1..N labels inside mask
    """
    if labels.size == 0:
        return labels
    lbl = labels.copy()
    sizes = np.bincount(lbl.ravel())
    small_ids = np.flatnonzero((sizes > 0) & (sizes < min_voxels))
    if len(small_ids) == 0:
        return lbl

    # 6-connected structure for dilation to find boundaries
    struct = ndi.generate_binary_structure(3, 1)

    for sid in small_ids:
        if sid == 0:
            continue
        mask = lbl == sid
        if not mask.any():
            continue
        if int(mask.sum()) >= min_voxels:
            continue
        # Border voxels (neighbors), collect neighbor ids & counts
        border = ndi.binary_dilation(mask, structure=struct) & (~mask)
        nbr_ids, counts = np.unique(lbl[border], return_counts=True)
        # exclude background and self
        valid = (nbr_ids != 0) & (nbr_ids != sid)
        if not valid.any():
            # Fallback: merge to the mode in a 2-voxel dilated ring
            border2 = ndi.binary_dilation(border, structure=struct) & (~mask)
            nbr_ids, counts = np.unique(lbl[border2], return_counts=True)
            valid = (nbr_ids != 0) & (nbr_ids != sid)
            if not valid.any():
                continue
        nbr_ids = nbr_ids[valid]
        counts = counts[valid]
        tgt = int(nbr_ids[np.argmax(counts)])
        lbl[mask] = tgt

    return lbl


def dice_large_regions(
    labels: np.ndarray,
    max_voxels: int,
) -> np.ndarray:
    """
    Force-split oversized labels into coarse geometric tiles.
    This intentionally prioritizes smaller pieces over anatomical boundaries.
    """
    if max_voxels <= 0 or labels.size == 0:
        return labels

    out = labels.copy()
    next_id = int(out.max())
    # Rough cube side for a target volume of max_voxels.
    tile_side = max(2, int(round(max_voxels ** (1.0 / 3.0))))

    region_ids = np.unique(out)
    region_ids = region_ids[region_ids != 0]
    for rid in region_ids:
        region = out == rid
        region_size = int(region.sum())
        if region_size <= max_voxels:
            continue

        coords = np.argwhere(region)
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        rel = coords - mins
        gx = (rel[:, 0] // tile_side).astype(np.int64)
        gy = (rel[:, 1] // tile_side).astype(np.int64)
        gz = (rel[:, 2] // tile_side).astype(np.int64)
        shape = (
            int(gx.max()) + 1,
            int(gy.max()) + 1,
            int(gz.max()) + 1,
        )
        tile_keys = np.ravel_multi_index((gx, gy, gz), dims=shape)
        _, tile_inverse = np.unique(tile_keys, return_inverse=True)

        out[region] = 0
        for tile_idx in range(int(tile_inverse.max()) + 1):
            tile_mask = tile_inverse == tile_idx
            if not tile_mask.any():
                continue
            next_id += 1
            vox = coords[tile_mask]
            out[vox[:, 0], vox[:, 1], vox[:, 2]] = next_id

    return out


def split_mask_into_supervoxels(
    mask: np.ndarray,
    target_voxels_per_sv: int = 25000,
    min_voxels: int = 2000,
    edge_cost: Optional[np.ndarray] = None,
    edge_weight: float = 0.7,
    enable_force_dicing: bool = False,
) -> np.ndarray:
    """
    Split a single object's boolean mask into supervoxels.
    Returns an int array with 0 outside mask, 1..K inside.
    If edge_cost is provided (e.g., gradient magnitude of raw image), the watershed
    cost is blended:  cost = edge_weight*edge_cost + (1-edge_weight)*(-distance)
    """
    mask = mask.astype(bool, copy=False)
    vox = int(np.count_nonzero(mask))
    if vox == 0:
        return np.zeros_like(mask, dtype=np.int32)

    # If already small, just one SV
    if vox <= target_voxels_per_sv:
        out = np.zeros_like(mask, dtype=np.int32)
        out[mask] = 1
        return out

    # Distance transform for seeds and geometry
    dist = ndi.distance_transform_edt(mask)

    # Decide how many seeds to place
    n_seeds = max(1, int(math.ceil(vox / float(target_voxels_per_sv))))

    # Heuristic spacing in voxels ~ cube root of target volume
    spacing = max(2, int(round((target_voxels_per_sv ** (1.0 / 3.0)) / 2.0)))

    # Seed coordinates from distance maxima
    coords = peak_local_max(
        dist,
        labels=mask,
        min_distance=spacing,
        num_peaks=n_seeds,
        exclude_border=False,
    )
    if coords.size == 0:
        coords = np.empty((0, mask.ndim), dtype=np.int32)
    else:
        coords = np.unique(coords.astype(np.int32, copy=False), axis=0)

    # Ensure every disconnected piece has at least one seed
    # Use bounding slices so we only inspect each connected component locally
    cc_labels, n_cc = ndi.label(
        mask, structure=ndi.generate_binary_structure(mask.ndim, 1)
    )
    cc_slices = ndi.find_objects(cc_labels)

    seed_list = [coords[i] for i in range(coords.shape[0])]
    seeded_components: set[int] = set()
    if coords.shape[0] > 0:
        seeded_components = {int(v) for v in cc_labels[tuple(coords.T)] if int(v) > 0}

    for cc_id in range(1, n_cc + 1):
        if cc_id in seeded_components:
            continue

        sl = cc_slices[cc_id - 1]
        if sl is None:
            continue

        cc_sub = (cc_labels[sl] == cc_id)
        if not cc_sub.any():
            continue

        dist_sub = dist[sl]
        # Pick the voxel with the largest distance inside this component.
        local_best = int(np.argmax(np.where(cc_sub, dist_sub, -np.inf)))
        local_idx = np.asarray(
            np.unravel_index(local_best, dist_sub.shape), dtype=np.int32
        )
        global_idx = local_idx + np.array([s.start for s in sl], dtype=np.int32)

        seed_list.append(global_idx)
        seeded_components.add(cc_id)

    markers = np.zeros_like(mask, dtype=np.int32)
    if seed_list:
        seed_coords = np.unique(np.vstack(seed_list).astype(np.int32), axis=0)
        markers[tuple(seed_coords.T)] = np.arange(
            1, seed_coords.shape[0] + 1, dtype=np.int32
        )

    # Build cost image
    if edge_cost is None:
        cost = -dist
    else:
        dnorm = dist / (dist.max() + 1e-6)
        enorm = edge_cost.astype(np.float32, copy=False)
        emx = float(enorm.max())
        if emx > 0:
            enorm = enorm / emx
        cost = edge_weight * enorm + (1.0 - edge_weight) * (1.0 - dnorm)

    labels = watershed(cost, markers=markers, mask=mask)

    # Fill any unassigned islands.
    missing = mask & (labels == 0)
    if missing.any():
        orphan_labels, n_orphans = ndi.label(
            missing, structure=ndi.generate_binary_structure(mask.ndim, 1)
        )
        next_label = int(labels.max())
        for orphan_id in range(1, n_orphans + 1):
            next_label += 1
            labels[orphan_labels == orphan_id] = next_label

    # Merge tiny supervoxels
    labels = merge_small_regions(labels, min_voxels=min_voxels)
    if enable_force_dicing:
        # Optional hard dicing mode for aggressive over-segmentation.
        labels = dice_large_regions(labels, max_voxels=target_voxels_per_sv)

    # Compact to 1..K
    labels, _, _ = relabel_sequential(labels)
    return labels


def process_chunk(
    seg_chunk: np.ndarray,
    raw_chunk: Optional[np.ndarray],
    chunk_index_xyz: Tuple[int, int, int],
    id_packer: GlobalIDPacker,
    target_voxels_per_sv: int = 25000,
    min_voxels_per_sv: int = 2000,
    edge_sigma: float = 1.5,
    enable_force_dicing: bool = False,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    seg_chunk: XYZ uint64 labels (0 = background) for THIS chunk only (no halo), shape (X, Y, Z).
    raw_chunk: optional XYZ raw intensities to guide splits via edges, shape (X, Y, Z).
    Returns (XYZ uint64 global IDs for this chunk, metadata dict with parent_sv_ids and sv_sizes).
    """
    assert seg_chunk.ndim == 3
    out = np.zeros_like(seg_chunk, dtype=np.uint64)

    # Edge cost from raw EM (optional).
    edge_cost = None
    if raw_chunk is not None:
        edge_cost = ndi.gaussian_gradient_magnitude(
            raw_chunk.astype(np.float32), sigma=edge_sigma
        )

    # Work label-by-label to ensure no SV crosses original object boundaries.
    labels_here = np.unique(seg_chunk)
    labels_here = labels_here[labels_here != 0]

    local_counter = 0
    cx, cy, cz = chunk_index_xyz
    parent_sv_ids: dict[int, list[int]] = {}  # parent_id -> list of global supervoxel IDs
    sv_sizes: list[int] = []  # size of each supervoxel in this chunk

    for lab in labels_here:
        parent_id = int(lab)
        mask = seg_chunk == lab
        # Decide whether to split further
        vox = int(mask.sum())
        if vox == 0:
            continue

        comp = split_mask_into_supervoxels(
            mask,
            target_voxels_per_sv=target_voxels_per_sv,
            min_voxels=min_voxels_per_sv,
            edge_cost=edge_cost,
            edge_weight=0.7 if edge_cost is not None else 0.0,
            enable_force_dicing=enable_force_dicing,
        )

        ncomp = int(comp.max())
        if ncomp == 0:
            continue

        gids = np.empty(ncomp + 1, dtype=np.uint64)  # 0 unused
        for k in range(1, ncomp + 1):
            local_counter += 1
            if local_counter > id_packer.max_local:
                raise RuntimeError(
                    f"Chunk {chunk_index_xyz} exceeded {id_packer.bl} local-id bits."
                )
            gids[k] = id_packer.encode(cx, cy, cz, local_counter)

        # Assign all voxels for this label at once
        out[mask] = gids[comp[mask]]

        # Sizes without per-component full scans
        sizes = np.bincount(comp[mask].ravel(), minlength=ncomp + 1)[1:]
        sv_sizes.extend(sizes.tolist())

        # Parent mapping
        parent_sv_ids.setdefault(parent_id, []).extend(gids[1:].astype(int).tolist())

    metadata = {
        "num_sv": local_counter,
        "parent_sv_ids": parent_sv_ids,  # parent_id -> list of global SV ids spawned from it
        "sv_sizes": sv_sizes,  # list of sizes
    }
    return out, metadata


def supervoxelize_array(
    seg: np.ndarray,
    raw: Optional[np.ndarray],
    chunk_xyz: Tuple[int, int, int] = (128, 128, 128),
    halo: int = 8,
    target_voxels_per_sv: int = 25000,
    min_voxels_per_sv: int = 2000,
    edge_sigma: float = 1.5,
    n_chunks_xyz: Optional[Tuple[int, int, int]] = None,
    min_local_bits: int = 20,
    enable_force_dicing: bool = False,
) -> Tuple[np.ndarray, list[dict[str, Any]]]:
    """
    seg: XYZ uint64 input segmentation (0=background), shape (X, Y, Z).
    raw: optional XYZ raw intensities to guide splitting, shape (X, Y, Z).
    Returns (XYZ uint64 array with global supervoxel IDs, list of per-chunk metadata).
    """
    assert seg.ndim == 3
    X, Y, Z = seg.shape
    out = np.zeros_like(seg, dtype=np.uint64)

    if n_chunks_xyz is None:
        n_chunks_xyz = chunk_grid_for_shape(seg.shape, chunk_xyz)

    # Respect the requested local bit allocation so ID packing matches caller expectations.
    id_packer = GlobalIDPacker(n_chunks_xyz, min_local_bits=min_local_bits)

    chunk_metadata_list: list[dict[str, Any]] = []

    for (cx, cy, cz), wslc, rslc in iter_chunk_bounds(seg.shape, chunk_xyz, halo=halo):
        seg_read = seg[rslc]
        raw_read = raw[rslc] if raw is not None else None

        # Crop to write region within the read chunk
        wx0 = wslc[0].start - rslc[0].start
        wy0 = wslc[1].start - rslc[1].start
        wz0 = wslc[2].start - rslc[2].start
        wx1 = wx0 + (wslc[0].stop - wslc[0].start)
        wy1 = wy0 + (wslc[1].stop - wslc[1].start)
        wz1 = wz0 + (wslc[2].stop - wslc[2].start)

        seg_chunk = seg_read[wx0:wx1, wy0:wy1, wz0:wz1]
        raw_chunk = (
            raw_read[wx0:wx1, wy0:wy1, wz0:wz1] if raw_read is not None else None
        )

        out_chunk, metadata = process_chunk(
            seg_chunk,
            raw_chunk,
            (cx, cy, cz),
            id_packer,
            target_voxels_per_sv=target_voxels_per_sv,
            min_voxels_per_sv=min_voxels_per_sv,
            edge_sigma=edge_sigma,
            enable_force_dicing=enable_force_dicing,
        )

        out[wslc] = out_chunk
        metadata["chunk_index"] = (cx, cy, cz)
        chunk_metadata_list.append(metadata)

    return out, chunk_metadata_list


def generate_supervoxel_tasks(
    graph_id: str,
    segmentation_channel: str,
    output_channel: str,
    raw_channel: str | None,
    mip: list | int,
    target_voxels_per_sv: int = 25000,
    min_voxels_per_sv: int = 2000,
    halo: int = 8,
    edge_sigma: float = 1.5,
    chunk_xyz: tuple = (128, 128, 128),
    z_start: int | None = None,
    z_end: int | None = None,
    bbox_min_xyz: tuple | None = None,
    bbox_max_xyz: tuple | None = None,
    enqueue_limit: int | None = None,
    min_local_bits: int = 20,
) -> Iterator[SupervoxelTaskPayload]:
    """
    Generate supervoxel tasks for chunks in a segmentation volume.

    Args:
        bbox_min_xyz: Optional (x_min, y_min, z_min) to constrain chunk generation.
        bbox_max_xyz: Optional (x_max, y_max, z_max) to constrain chunk generation.
        z_start, z_end: Deprecated; use bbox_min_xyz/bbox_max_xyz instead.

    Yields SupervoxelTaskPayload for each chunk.
    """
    seg_data = CloudVolume(
        segmentation_channel, mip=cast(Any, mip), cache=True, use_https=True,
    )
    volume_chunk_xyz = tuple(int(c) for c in tuple(getattr(seg_data, "chunk_size"))[:3])
    # if tuple(chunk_xyz) != volume_chunk_xyz:
    #     print(f"[supervoxel] Adjusting chunk size to volume chunk size {volume_chunk_xyz} (was {chunk_xyz})")
    #     chunk_xyz = volume_chunk_xyz

    voxel_offset_raw = getattr(seg_data, "voxel_offset")
    voxel_offset = tuple(int(v) for v in tuple(voxel_offset_raw)[:3])
    shape_raw = getattr(seg_data, "shape")
    volume_shape_xyz = tuple(int(s) for s in tuple(shape_raw)[:3])

    # Default bounds: entire volume
    x_start = voxel_offset[0]
    x_stop = voxel_offset[0] + volume_shape_xyz[0]
    y_start = voxel_offset[1]
    y_stop = voxel_offset[1] + volume_shape_xyz[1]
    z_start_voxel = 0
    z_end_voxel = volume_shape_xyz[2]

    # Apply bounding box if provided
    if bbox_min_xyz is not None:
        x_start = max(x_start, bbox_min_xyz[0])
        y_start = max(y_start, bbox_min_xyz[1])
        z_start_voxel = max(z_start_voxel, bbox_min_xyz[2] - voxel_offset[2])

    if bbox_max_xyz is not None:
        x_stop = min(x_stop, bbox_max_xyz[0])
        y_stop = min(y_stop, bbox_max_xyz[1])
        z_end_voxel = min(z_end_voxel, bbox_max_xyz[2] - voxel_offset[2])

    # Fallback to z_start/z_end if bbox not provided
    if bbox_min_xyz is None and z_start is not None:
        z_start_voxel = max(0, min(volume_shape_xyz[2], z_start))

    if bbox_max_xyz is None and z_end is not None:
        z_end_voxel = max(z_start_voxel, min(volume_shape_xyz[2], z_end))

    z_start = voxel_offset[2] + z_start_voxel
    z_stop = voxel_offset[2] + z_end_voxel

    # Expand requested bounds to the chunk grid to keep output writes aligned.
    def _align_down(v: int, offset: int, size: int) -> int:
        return offset + ((v - offset) // size) * size

    def _align_up(v: int, offset: int, size: int) -> int:
        return offset + int(math.ceil((v - offset) / float(size))) * size

    x_start = max(voxel_offset[0], _align_down(int(x_start), voxel_offset[0], chunk_xyz[0]))
    y_start = max(voxel_offset[1], _align_down(int(y_start), voxel_offset[1], chunk_xyz[1]))
    z_start = max(voxel_offset[2], _align_down(int(z_start), voxel_offset[2], chunk_xyz[2]))

    x_stop = min(
        voxel_offset[0] + volume_shape_xyz[0],
        _align_up(int(x_stop), voxel_offset[0], chunk_xyz[0]),
    )
    y_stop = min(
        voxel_offset[1] + volume_shape_xyz[1],
        _align_up(int(y_stop), voxel_offset[1], chunk_xyz[1]),
    )
    z_stop = min(
        voxel_offset[2] + volume_shape_xyz[2],
        _align_up(int(z_stop), voxel_offset[2], chunk_xyz[2]),
    )

    # Calculate chunk grid for the entire volume
    n_chunks_xyz = chunk_grid_for_shape(volume_shape_xyz, chunk_xyz)

    from intern.utils.parallel import block_compute

    blocks = list(
        block_compute(
            x_start=x_start,
            x_stop=x_stop,
            y_start=y_start,
            y_stop=y_stop,
            z_start=z_start,
            z_stop=z_stop,
            block_size=chunk_xyz,
        )
    )

    if enqueue_limit:
        print(f"Queueing {min(len(blocks), enqueue_limit)} supervoxel chunks")
    else:
        print(f"Queueing {len(blocks)} supervoxel chunks")

    for i, ((x_start_block, x_stop_block), (y_start_block, y_stop_block), (z_start_block, z_stop_block)) in enumerate(
        blocks
    ):
        if enqueue_limit is not None and i >= enqueue_limit:
            break

        # Compute chunk index from block bounds
        cx = (x_start_block - voxel_offset[0]) // chunk_xyz[0]
        cy = (y_start_block - voxel_offset[1]) // chunk_xyz[1]
        cz = (z_start_block - voxel_offset[2]) // chunk_xyz[2]

        payload: SupervoxelTaskPayload = {
            "graph_id": graph_id,
            "task_type": "supervoxel",
            "cuboid_start": (x_start_block, y_start_block, z_start_block),
            "cuboid_radius": (
                x_stop_block - x_start_block,
                y_stop_block - y_start_block,
                z_stop_block - z_start_block,
            ),
            "chunk_index_xyz": (cx, cy, cz),
            "n_chunks_xyz": n_chunks_xyz,
            "segmentation_channel": segmentation_channel,
            "output_channel": output_channel,
            "raw_channel": raw_channel,
            "mip": mip,
            "target_voxels_per_sv": target_voxels_per_sv,
            "min_voxels_per_sv": min_voxels_per_sv,
            "halo": halo,
            "edge_sigma": edge_sigma,
            "min_local_bits": min_local_bits,
        }
        yield payload
