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

from database import SupervoxelTaskPayload

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
    shape_zyx: Tuple[int, int, int], chunk_xyz: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """Returns (n_chunks_x, n_chunks_y, n_chunks_z) given array shape in ZYX and chunk in XYZ."""
    Z, Y, X = shape_zyx
    cx, cy, cz = chunk_xyz  # user provided XYZ
    # Convert to ZYX for math
    nz = math.ceil(Z / cz)
    ny = math.ceil(Y / cy)
    nx = math.ceil(X / cx)
    return (nx, ny, nz)


def iter_chunk_bounds(
    shape_zyx: Tuple[int, int, int],
    chunk_xyz: Tuple[int, int, int],
    halo: int = 0,
) -> Iterator[Tuple[Tuple[int, int, int], Tuple[slice, slice, slice], Tuple[slice, slice, slice]]]:
    """
    Yield per-chunk (chunk_index_xyz, write_slices_zyx, read_slices_zyx).
    read_slices includes halo; write_slices excludes halo (hard boundary).
    """
    Z, Y, X = shape_zyx
    cx, cy, cz = chunk_xyz
    # step in XYZ, translate to ZYX indexing
    stepZ, stepY, stepX = cz, cy, cx
    for cz_i, z0 in enumerate(range(0, Z, stepZ)):
        for cy_i, y0 in enumerate(range(0, Y, stepY)):
            for cx_i, x0 in enumerate(range(0, X, stepX)):
                z1 = min(z0 + stepZ, Z)
                y1 = min(y0 + stepY, Y)
                x1 = min(x0 + stepX, X)

                # write (no halo)
                wslc = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

                # read (with halo, clipped to volume)
                rz0 = max(0, z0 - halo)
                ry0 = max(0, y0 - halo)
                rx0 = max(0, x0 - halo)
                rz1 = min(Z, z1 + halo)
                ry1 = min(Y, y1 + halo)
                rx1 = min(X, x1 + halo)
                rslc = (slice(rz0, ry1), slice(ry0, ry1), slice(rx0, rx1))

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


def split_mask_into_supervoxels(
    mask: np.ndarray,
    target_voxels_per_sv: int = 25000,
    min_voxels: int = 2000,
    edge_cost: Optional[np.ndarray] = None,
    edge_weight: float = 0.7,
) -> np.ndarray:
    """
    Split a single object's boolean mask into supervoxels.
    Returns an int array with 0 outside mask, 1..K inside.
    If edge_cost is provided (e.g., gradient magnitude of raw image), the watershed
    cost is blended:  cost = edge_weight*edge_cost + (1-edge_weight)*(-distance)
    """
    vox = int(mask.sum())
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

    # Seed coordinates from distance maxima (limited to n_seeds)
    coords = peak_local_max(
        dist,
        labels=mask,
        min_distance=spacing,
        num_peaks=n_seeds,
        exclude_border=False,
    )

    if coords.size == 0:
        # fallback: single seed at the maximal distance point
        maxpos = np.unravel_index(np.argmax(dist), dist.shape)
        coords = np.array([maxpos], dtype=int)

    markers = np.zeros_like(mask, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[z, y, x] = i

    # Build cost image
    if edge_cost is None:
        cost = -dist
    else:
        # Normalize both terms to [0,1] to keep units compatible
        dnorm = dist / (dist.max() + 1e-6)
        enorm = edge_cost.astype(np.float32)
        emx = enorm.max()
        if emx > 0:
            enorm = enorm / emx
        cost = edge_weight * enorm + (1.0 - edge_weight) * (1.0 - dnorm)  # low inside, high at edges

    labels = watershed(cost, markers=markers, mask=mask)

    # Merge tiny supervoxels
    labels = merge_small_regions(labels, min_voxels=min_voxels)

    # Compact to 1..K
    labels, _, _ = relabel_sequential(labels)
    return labels


def process_chunk(
    seg_chunk: np.ndarray,
    raw_chunk: np.ndarray,
    chunk_index_xyz: Tuple[int, int, int],
    id_packer: GlobalIDPacker,
    target_voxels_per_sv: int = 25000,
    min_voxels_per_sv: int = 2000,
    edge_sigma: float = 1.5,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    seg_chunk: ZYX uint64 labels (0 = background) for THIS chunk only (no halo).
    raw_chunk: ZYX raw intensities to guide splits via edges.
    Returns (ZYX uint64 global IDs for this chunk, metadata dict with parent_counts and sv_sizes).
    """
    assert seg_chunk.ndim == 3
    out = np.zeros_like(seg_chunk, dtype=np.uint64)

    # Edge cost from raw EM
    edge_cost = ndi.gaussian_gradient_magnitude(raw_chunk.astype(np.float32), sigma=edge_sigma)

    # Work label-by-label to ensure no SV crosses original object boundaries.
    labels_here = np.unique(seg_chunk)
    labels_here = labels_here[labels_here != 0]

    local_counter = 0
    cx, cy, cz = chunk_index_xyz
    parent_sv_ids: dict[int, list[int]] = {}  # parent_id -> list of global supervoxel IDs
    sv_sizes: list[int] = []  # size of each supervoxel in this chunk

    for lab in labels_here:
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
            edge_weight=0.7,
        )

        ncomp = int(comp.max())
        if ncomp == 0:
            continue

        # Track this parent segment
        if lab not in parent_sv_ids:
            parent_sv_ids[lab] = []

        # Assign global IDs
        for k in range(1, ncomp + 1):
            local_counter += 1
            if local_counter > id_packer.max_local:
                raise RuntimeError(
                    f"Exceeded local_id capacity ({id_packer.max_local}) for chunk {chunk_index_xyz}. "
                    "Increase target_voxels_per_sv or allocate more local bits."
                )
            gid = id_packer.encode(cx, cy, cz, local_counter)
            sv_mask = comp == k
            out[sv_mask] = gid
            sv_sizes.append(int(sv_mask.sum()))
            parent_sv_ids[lab].append(int(gid))

    metadata = {
        "num_sv": local_counter,
        "parent_sv_ids": parent_sv_ids,  # parent_id -> list of global SV ids spawned from it
        "sv_sizes": sv_sizes,  # list of sizes
    }
    return out, metadata


def supervoxelize_array(
    seg: np.ndarray,
    raw: np.ndarray,
    chunk_xyz: Tuple[int, int, int] = (128, 128, 128),
    halo: int = 8,
    target_voxels_per_sv: int = 25000,
    min_voxels_per_sv: int = 2000,
    edge_sigma: float = 1.5,
    n_chunks_xyz: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, list[dict[str, Any]]]:
    """
    seg: ZYX uint64 input segmentation (0=background).
    raw: ZYX raw intensities to guide splitting.
    Returns (ZYX uint64 array with global supervoxel IDs, list of per-chunk metadata).
    """
    assert seg.ndim == 3
    Z, Y, X = seg.shape
    out = np.zeros_like(seg, dtype=np.uint64)

    if n_chunks_xyz is None:
        n_chunks_xyz = chunk_grid_for_shape(seg.shape, chunk_xyz)

    id_packer = GlobalIDPacker(n_chunks_xyz, min_local_bits=20)

    chunk_metadata_list: list[dict[str, Any]] = []

    for (cx, cy, cz), wslc, rslc in iter_chunk_bounds(seg.shape, chunk_xyz, halo=halo):
        seg_read = seg[rslc]
        raw_read = raw[rslc]

        # Crop to write region within the read chunk
        wz0 = wslc[0].start - rslc[0].start
        wy0 = wslc[1].start - rslc[1].start
        wx0 = wslc[2].start - rslc[2].start
        wz1 = wz0 + (wslc[0].stop - wslc[0].start)
        wy1 = wy0 + (wslc[1].stop - wslc[1].start)
        wx1 = wx0 + (wslc[2].stop - wslc[2].start)

        seg_chunk = seg_read[wz0:wz1, wy0:wy1, wx0:wx1]
        raw_chunk = raw_read[wz0:wz1, wy0:wy1, wx0:wx1]

        out_chunk, metadata = process_chunk(
            seg_chunk,
            raw_chunk,
            (cx, cy, cz),
            id_packer,
            target_voxels_per_sv=target_voxels_per_sv,
            min_voxels_per_sv=min_voxels_per_sv,
            edge_sigma=edge_sigma,
        )

        out[wslc] = out_chunk
        metadata["chunk_index"] = (cx, cy, cz)
        chunk_metadata_list.append(metadata)

    return out, chunk_metadata_list


def generate_supervoxel_tasks(
    graph_id: str,
    segmentation_channel: str,
    output_channel: str,
    raw_channel: str,
    mip: list | int,
    target_voxels_per_sv: int = 25000,
    min_voxels_per_sv: int = 2000,
    halo: int = 8,
    edge_sigma: float = 1.5,
    chunk_xyz: tuple = (128, 128, 128),
    z_start: int | None = None,
    z_end: int | None = None,
    enqueue_limit: int | None = None,
    min_local_bits: int = 20,
) -> Iterator[SupervoxelTaskPayload]:
    """
    Generate supervoxel tasks for all chunks in a segmentation volume.
    Yields SupervoxelTaskPayload for each chunk.
    """
    seg_data = CloudVolume(
        segmentation_channel, mip=cast(Any, mip), cache=True, use_https=True,
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

    # Calculate chunk grid for the entire volume
    n_chunks_xyz = chunk_grid_for_shape(volume_shape, chunk_xyz)
    cx_max, cy_max, cz_max = n_chunks_xyz

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
