import argparse


def _parse_mip_argument(mip_arg: str) -> list | int:
    # Parse MIP from string - always convert to a list for consistency
    if "," in mip_arg:
        mip = [int(x) for x in mip_arg.split(",")]
        return mip
    else:
        try:
            # If a single value, make it a list with the same value for all dimensions
            single_mip = int(mip_arg)
            mip = [single_mip, single_mip, single_mip]
            return mip
        except ValueError:
            print(
                f"Error: MIP value '{mip_arg}' is not valid. Use a single integer or comma-separated integers."
            )
            exit(1)


def _attach_global_arguments(
    parser: argparse.ArgumentParser,
    queue_url_default: str = "https://sqs.us-east-1.amazonaws.com/407510763690/CloudomeJobs",
):
    parser.add_argument(
        "--mip",
        type=str,
        default="72,72,84",
        help="MIP value as either a single int or comma-separated values (e.g., 72,72,84)",
    )
    parser.add_argument(
        "--queue-url",
        type=str,
        default=queue_url_default,
        help="Queue URL (SQS URL or FQ file path) for job queue",
    )
    return parser


def _attach_contactome_parser(subparsers: argparse._SubParsersAction):
    contactome_parser = subparsers.add_parser(
        "contactome", help="Commands related to contactome"
    )
    contactome_subparsers = contactome_parser.add_subparsers(
        dest="command", required=True
    )
    contactome_generate_parser = contactome_subparsers.add_parser(
        "generate", help="Generate cuboidwise tasks for contactome"
    )
    contactome_generate_parser.add_argument(
        "--graph-id", type=str, required=True, help="Graph ID for processing"
    )
    contactome_generate_parser.add_argument(
        "--segmentation-channel",
        type=str,
        required=True,
        help="S3 path to segmentation channel data",
    )
    contactome_generate_parser.add_argument(
        "--block-size-x", type=int, default=64, help="Block size for X dimension"
    )
    contactome_generate_parser.add_argument(
        "--block-size-y", type=int, default=64, help="Block size for Y dimension"
    )
    contactome_generate_parser.add_argument(
        "--block-size-z", type=int, default=32, help="Block size for Z dimension"
    )
    contactome_generate_parser.add_argument(
        "--z-start", type=int, default=None, help="Starting Z slice"
    )
    contactome_generate_parser.add_argument(
        "--z-end", type=int, default=None, help="Ending Z slice"
    )
    contactome_generate_parser.add_argument(
        "--enqueue-limit",
        type=int,
        default=None,
        help="Limit the number of tasks to enqueue",
    )
    return contactome_parser, contactome_subparsers, (contactome_generate_parser,)


def _attach_synapse_parser(subparsers: argparse._SubParsersAction):
    synapses_parser = subparsers.add_parser(
        "synapses", help="Commands related to synapses"
    )
    synapses_subparsers = synapses_parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate (synapses)
    syn_generate_parser = synapses_subparsers.add_parser(
        "generate", help="Generate centroids for synapse mask"
    )
    syn_generate_parser.add_argument(
        "--synapse-channel",
        type=str,
        required=True,
        help="S3 path to synapse channel data",
    )
    syn_generate_parser.add_argument(
        "--output-file",
        type=str,
        default="centroids.csv",
        help="Output file path for centroids",
    )

    # Subcommand: enqueue (synapses)
    enqueue_parser = synapses_subparsers.add_parser(
        "enqueue", help="Enqueue centroids from file"
    )
    enqueue_parser.add_argument(
        "--graph-id", type=str, required=True, help="Graph ID for processing"
    )
    enqueue_parser.add_argument(
        "--centroids-file", type=str, required=True, help="Path to centroids file"
    )
    enqueue_parser.add_argument(
        "--synapse-channel",
        type=str,
        required=True,
        help="S3 path to synapse channel data",
    )
    enqueue_parser.add_argument(
        "--segmentation-channel",
        type=str,
        required=True,
        help="S3 path to segmentation channel data",
    )
    enqueue_parser.add_argument(
        "--enqueue-limit",
        type=int,
        default=None,
        help="Limit the number of centroids to enqueue",
    )

    return synapses_parser, synapses_subparsers, (syn_generate_parser, enqueue_parser)


__all__ = [
    "_parse_mip_argument",
    "_attach_global_arguments",
    "_attach_contactome_parser",
    "_attach_synapse_parser",
]
