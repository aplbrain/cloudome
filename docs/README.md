# Getting Started

Cloudome is a command line tool. To get started, you will need to [install uv](https://docs.astral.sh/uv/getting-started/installation/) and then run `uv sync` to install the dependencies.

From there, you should head to the [AWS docs](docs/AWS.md) or the [local docs](docs/Local.md) for instructions on running Cloudome using the parallelizable compute available to you. We recommend running connectomes on AWS and chunk-wise calculations on an HPC cluster, as connectome subtasks are very small and therefore good candidates for extreme parallelization. Chunk-wise calculations will become more efficient as chunk size increases and are therefore better candidates for an HPC cluster.

The AWS guide uses Zappa with the AWS tools SQS, Lambda, and DynamoDB to complete a job. Results can be downloaded from DynamoDB as a CSV. With this setup, connectomes cost about $1/25k synapses to compute, and execution time is limited only by how quickly jobs can be queued from the local machine.

The local guide contains instructions on completing a job using a local HPC cluster. Results are pushed to a SQLite DB, with an optional script to export as CSV.
