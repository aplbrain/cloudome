#!/usr/bin/env python3
"""
Create a CloudVolume layer based on an existing segmentation layer.

This script creates a new CloudVolume layer with identical properties (offset, size,
resolution, encoding, etc.) as the source segmentation layer. Useful for preparing
output layers for supervoxel generation or other processing pipelines.

Usage:
    python create_sv_output_layer.py \
        --input-channel s3://bucket/path/to/seg/ \
        --output-channel s3://bucket/path/to/output/ \
        --mip 1 
"""

import argparse
from typing import List, Tuple
import numpy as np
from cloudvolume import CloudVolume


def create_output_layer(
    input_channel: str,
    output_channel: str,
    mip: List | int = 0,
    data_type: str = "uint64",
    chunk_size: Tuple[int, int, int] | None = None,
) -> None:
    """
    Create a new CloudVolume layer matching the input layer's properties.

    Args:
        input_channel: Source CloudVolume path (e.g., s3://bucket/path/to/seg/)
        output_channel: Destination CloudVolume path to create
        mip: Resolution level (scalar or list [x, y, z])
        data_type: Data type for output (default: uint64)
        chunk_size: Chunk size (x, y, z). If None, uses input layer's chunk size.
    """
    print(f"Reading metadata from: {input_channel}")
    input_cv = CloudVolume(
        input_channel,
        mip=mip,
        cache=False,
        use_https=True,
    )

    # Extract properties from input layer
    num_channels = input_cv.num_channels
    voxel_offset = input_cv.voxel_offset
    volume_shape = input_cv.shape[:-1]  
    resolution = input_cv.resolution
    encoding = input_cv.encoding if hasattr(input_cv, "encoding") else "raw"
    
    if chunk_size is None:
        chunk_size = input_cv.chunk_size


    print(f"\nInput Layer Properties:")
    print(f"  Number of Channels: {num_channels}")
    print(f"  Voxel Offset: {voxel_offset}")
    print(f"  Shape: {volume_shape}")
    print(f"  Resolution: {resolution}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Encoding: {encoding}")

    print(f"\nCreating output layer: {output_channel}")
    info = CloudVolume.create_new_info(
        layer_type="segmentation",
        num_channels=num_channels,
        resolution=resolution,
        voxel_offset=voxel_offset,
        volume_size=volume_shape,
        chunk_size=chunk_size,
        encoding=encoding,
        data_type=data_type,
    )
    vol = CloudVolume(output_channel, info=info)
    vol.commit_info()

    print(f"✓ Output layer created successfully!")
    print(f"  Location: {output_channel}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a CloudVolume layer with properties matching an input segmentation layer."
    )
    parser.add_argument(
        "--input-channel",
        type=str,
        required=True,
        help="S3 path to input segmentation layer (e.g., s3://bucket/path/to/seg/)",
    )
    parser.add_argument(
        "--output-channel",
        type=str,
        required=True,
        help="S3 path for output layer to create (e.g., s3://bucket/path/to/output/)",
    )
    parser.add_argument(
        "--mip",
        type=str,
        default="0",
        help="MIP level as single int or comma-separated floats",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="uint64",
        help="Data type for output layer (default: uint64)",
    )
    parser.add_argument(
        "--chunk-size",
        type=str,
        default=None,
        help="Chunk size as comma-separated values (e.g., 128,128,128). If not provided, uses input layer's chunk size.",
    )

    args = parser.parse_args()

    # Parse MIP argument
    if "," in args.mip:
        mip = [float(x) for x in args.mip.split(",")]
    else:
        mip = int(args.mip)

    # Parse chunk size if provided
    chunk_size = None
    if args.chunk_size:
        chunk_size = tuple(int(x) for x in args.chunk_size.split(","))

    create_output_layer(
        input_channel=args.input_channel,
        output_channel=args.output_channel,
        mip=mip,
        data_type=args.data_type,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    main()
