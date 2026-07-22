# Copyright © 2026 The Johns Hopkins University Applied Physics Laboratory LLC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from manage import (
    generate_cuboidwise_tasks_for_contactome_or_volume,
    enqueue_centroids_from_file,
    generate_supervoxel_tasks_for_queue,
    export_dynamodb_results_to_csv,
)
from cloudome.database import (
    SynapseEdgeResultsModel,
    SynapseEdgeTaskPayload,
    ContactomeEdgeTaskPayload,
    ContactEdgeResultsModel,
    TaskType,
    VolumeTaskPayload,
    SupervoxelTaskPayload,
)
import time
import boto3

block_size = (64, 64, 32)
queue_url = "https://sqs.us-east-1.amazonaws.com/<your-sqs-queue-id>/<your-sqs-queue-name>"
segmentation_channel = "s3://path/to/segmentation/"
synapse_channel = "s3://path/to/synapses/"
raw_channel = "s3://path/to/images/"
output_channel = "s3://path/to/supervoxel/output/"
centroids_file = "path/to/synapse-centroids.csv"
mip = [36, 36, 42]
z_start = 1800
z_end = 1810
# The below will change depending on above input variables
correct_num_lines_for_contactome_task = 445
correct_num_lines_for_volume_task = 432
correct_num_lines_for_connectome_task = 100

def test_enqueue_contactome_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-contactome-{now}"

    print(f"Initiating test {graph_id}")
    generate_cuboidwise_tasks_for_contactome_or_volume(
        sqs_url=queue_url,
        graph_id=graph_id,
        task_type="contactome",
        segmentation_channel=segmentation_channel,
        mip=mip,
        block_size=block_size,
        z_start=z_start,
        z_end=z_end,
        enqueue_limit=100,
    )
    return graph_id

def test_enqueue_volume_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-volume-{now}"

    print(f"Initiating test {graph_id}")
    generate_cuboidwise_tasks_for_contactome_or_volume(
        sqs_url=queue_url,
        graph_id=graph_id,
        task_type="volume",
        segmentation_channel=segmentation_channel,
        mip=mip,
        block_size=block_size,
        z_start=z_start,
        z_end=z_end,
        enqueue_limit=100,
    )
    return graph_id

def test_enqueue_connectome_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-connectome-{now}"

    print(f"Initiating test {graph_id}")
    enqueue_centroids_from_file(
        sqs_url=queue_url,
        graph_id=graph_id,
        filename=centroids_file,
        synapse_channel=synapse_channel,
        segmentation_channel=segmentation_channel,
        mip=mip,
        enqueue_limit=100,
    )
    return graph_id


def test_enqueue_supervoxel_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-supervoxel-{now}"

    print(f"Initiating test {graph_id}")
    chunk_size = (128, 128, 128)
    generate_supervoxel_tasks_for_queue(
        sqs_url=queue_url,
        graph_id=graph_id,
        segmentation_channel=segmentation_channel,
        output_channel=output_channel,
        raw_channel=raw_channel,
        mip=mip,
        target_voxels_per_sv=25000,
        min_voxels_per_sv=2000,
        halo=8,
        edge_sigma=1.5,
        chunk_xyz=chunk_size,
        z_start=z_start,
        z_end=z_end,
        enqueue_limit=10,
    )
    return graph_id

def wait_for_queue_empty():
    sqs = boto3.client("sqs", region_name="us-east-1")
    approx_not_visible = -1
    while approx_not_visible != "0":
        time.sleep(2)
        response = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=["ApproximateNumberOfMessagesNotVisible"]
        )
        approx_not_visible = response["Attributes"]["ApproximateNumberOfMessagesNotVisible"]
        print(f"Messages left in queue: {approx_not_visible}")
    return

def test_results(graph_id, correct_len):

    wait_for_queue_empty()
    count = -1
    max_tries = 5
    num_tries = 0
    while count != correct_len and num_tries < max_tries:
        count = ContactEdgeResultsModel.count(graph_id)
        print(f"Results completed: {count}")
        num_tries += 1
        time.sleep(2)

    if num_tries >= max_tries:
        print(f"Test {graph_id} failed")
    else:
        print(f"Test {graph_id} successful")


if __name__ == "__main__":

    contactome_graph_id = test_enqueue_contactome_tasks()
    time.sleep(2)
    test_results(contactome_graph_id, correct_num_lines_for_contactome_task)

    volume_graph_id = test_enqueue_volume_tasks()
    time.sleep(2)
    test_results(volume_graph_id, correct_num_lines_for_volume_task)

    connectome_graph_id = test_enqueue_connectome_tasks()
    time.sleep(2)
    test_results(connectome_graph_id, correct_num_lines_for_connectome_task)

    supervoxel_graph_id = test_enqueue_supervoxel_tasks()
    time.sleep(2)
    # Supervoxel tasks write directly to output layer, check via queue monitoring
    wait_for_queue_empty()
    print(f"Supervoxel test {supervoxel_graph_id} enqueued and processed")
    time.sleep(2)
    test_results(connectome_graph_id, correct_num_lines_for_connectome_task)
