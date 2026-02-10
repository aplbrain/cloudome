from local_manage import (
    provision_db_contactome,
    provision_db_synapses,
    provision_db_supervoxel,
    enqueue_cuboidwise_tasks_for_contactome_or_volume,
    enqueue_centroids_from_file,
    enqueue_supervoxel_tasks,
    run_worker
)
from database import (
    SynapseEdgeResultsModel,
    SynapseEdgeTaskPayload,
    ContactomeEdgeTaskPayload,
    ContactEdgeResultsModel,
    TaskType,
    VolumeTaskPayload,
    SupervoxelTaskPayload,
)
import time
from taskqueue import TaskQueue
import sqlite3

# If you change these params, you must also change lines 25+26
block_size = (64, 64, 32)
queue_url = "fq://q-CloudomeTasks"
sqlite_db_path= "cloudome-results.db"
segmentation_channel = "s3://cvdb-bossdb-boss/smith2024/zebrafish/agglomeration_checkpoint_40000/"
synapse_channel = "s3://cvdb-bossdb-boss/smith2024/zebrafish/synapses/"
raw_channel = "s3://cvdb-bossdb-boss/smith2024/zebrafish/em"
output_sv_channel = "s3://cvdb-bossdb-boss/smith2024/zebrafish/supervoxels_agg_chkpt_40000/"
centroids_file = "test/synapse-centroids.csv"
mip = [36, 36, 42]
z_start = 1800
z_end = 1810

correct_num_lines_for_contactome_task = 445
correct_num_lines_for_volume_task = 432
correct_num_lines_for_connectome_task = 100
correct_num_lines_for_supervoxel_task = 10  # 10 chunks processed

def test_enqueue_contactome_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-contactome-{now}"
        
    print(f"Initiating test {graph_id}")
    enqueue_cuboidwise_tasks_for_contactome_or_volume(
        fq_url=queue_url,
        graph_id=graph_id,
        task_type="contactome",
        segmentation_channel=segmentation_channel,
        sqlite_db_path=sqlite_db_path,
        mip=mip,
        block_size=block_size,
        z_start=z_start,
        z_end=z_end,
        enqueue_limit=100,
    )
    run_worker(queue_url, max_tasks=100, verbose=True)
    return graph_id
    
def test_enqueue_volume_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-volume-{now}"
        
    print(f"Initiating test {graph_id}")
    enqueue_cuboidwise_tasks_for_contactome_or_volume(
        fq_url=queue_url,
        graph_id=graph_id,
        task_type="volume",
        segmentation_channel=segmentation_channel,
        sqlite_db_path=sqlite_db_path,
        mip=mip,
        block_size=block_size,
        z_start=z_start,
        z_end=z_end,
        enqueue_limit=100,
    )
    run_worker(queue_url, max_tasks=100, verbose=True)
    return graph_id

def test_enqueue_connectome_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-connectome-{now}"

    print(f"Initiating test {graph_id}")
    enqueue_centroids_from_file(
        fq_url=queue_url,
        graph_id=graph_id,
        filename=centroids_file,
        sqlite_db_path=sqlite_db_path,
        synapse_channel=synapse_channel,
        segmentation_channel=segmentation_channel,
        mip=mip,
        enqueue_limit=100,
    )
    run_worker(queue_url, max_tasks=100, verbose=True)
    return graph_id


def test_enqueue_supervoxel_tasks(graph_id=None):
    if not graph_id:
        now = int(time.time())
        graph_id = f"test-supervoxel-{now}"

    print(f"Initiating test {graph_id}")
    chunk_size = (128, 128, 128)
    enqueue_supervoxel_tasks(
        fq_url=queue_url,
        graph_id=graph_id,
        segmentation_channel=segmentation_channel,
        output_channel=output_sv_channel,
        raw_channel=raw_channel,
        sqlite_db_path=sqlite_db_path,
        mip=0,
        target_voxels_per_sv=25000,
        min_voxels_per_sv=2000,
        halo=8,
        edge_sigma=1.5,
        chunk_xyz=chunk_size,
        z_start=z_start,
        z_end=z_end,
        enqueue_limit=10,
    )
    run_worker(queue_url, max_tasks=10, verbose=True)
    return graph_id

def wait_for_queue_empty():
    tq = TaskQueue(queue_url)
    remaining = -1
    while remaining != 0:
        time.sleep(2)
        remaining = tq.inserted - tq.completed
        print("Approx tasks remaining:", remaining)
    return

def test_results(table_name, graph_id, correct_len):

    wait_for_queue_empty()

    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT COUNT(*) AS count
        FROM '{table_name}'
        WHERE graph_id = '{graph_id}'
    """)
    count = cursor.fetchone()[0]
    print(f"Results completed: {count}")

    conn.close()
    if count != correct_len:
        print(f"Test {graph_id} failed")
    else:
        print(f"Test {graph_id} successful")
    

if __name__ == "__main__":

    # provision_db_contactome(sqlite_db_path)
    # provision_db_synapses(sqlite_db_path)
    provision_db_supervoxel(sqlite_db_path)
    
    # contactome_graph_id = test_enqueue_contactome_tasks()
    # test_results("contactome_edges", contactome_graph_id, correct_num_lines_for_contactome_task)
    
    # volume_graph_id = test_enqueue_volume_tasks()
    # test_results("volume_counts", volume_graph_id, correct_num_lines_for_volume_task)

    # connectome_graph_id = test_enqueue_connectome_tasks()
    # test_results("synapse_edges", connectome_graph_id, correct_num_lines_for_connectome_task)

    supervoxel_graph_id = test_enqueue_supervoxel_tasks()
    test_results("supervoxel_chunks", supervoxel_graph_id, correct_num_lines_for_supervoxel_task)
