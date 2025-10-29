from __future__ import annotations

from functools import partial
from typing import Iterable

from taskqueue import queueable

from .processing import return_ctc_edges, return_seg_edge, return_volume_counts
from .storage import ResultRecord, get_result_store


@queueable
def process_task(payload: dict, store_backend: str = "dynamodb", store_kwargs: dict | None = None) -> None:
    """Queueable function executed by ptq workers."""

    store = get_result_store(store_backend, **(store_kwargs or {}))
    task_type = payload.get("task_type", "synapse")
    graph_id = payload.get("graph_id")

    records: list[ResultRecord] = []

    if task_type == "contactome":
        edges = return_ctc_edges(payload)
        if graph_id is None:
            raise ValueError("contactome payload missing graph_id")
        for (pre_id, post_id), weight in edges.items():
            records.append(
                ResultRecord(
                    result_type="contactome",
                    graph_id=graph_id,
                    payload=(
                        f"ctc_x{payload['cuboid_start'][0]}_y{payload['cuboid_start'][1]}_"
                        f"z{payload['cuboid_start'][2]}_pre{pre_id}_post{post_id}_w{int(weight)}"
                    ),
                )
            )
    elif task_type == "volume":
        counts = return_volume_counts(payload)
        if graph_id is None:
            raise ValueError("volume payload missing graph_id")
        for seg_id, count in counts.items():
            records.append(
                ResultRecord(
                    result_type="volume",
                    graph_id=graph_id,
                    payload=(
                        f"vol_x{payload['cuboid_start'][0]}_y{payload['cuboid_start'][1]}_"
                        f"z{payload['cuboid_start'][2]}_seg{seg_id}_v{int(count)}"
                    ),
                )
            )
    else:
        # default to synapse edge processing
        if graph_id is None:
            raise ValueError("synapse payload missing graph_id")
        pre_id, post_id = return_seg_edge(payload)
        records.append(
            ResultRecord(
                result_type="synapse",
                graph_id=graph_id,
                payload=(
                    f"syn_x{payload['centroid_xyz'][0]}_y{payload['centroid_xyz'][1]}_"
                    f"z{payload['centroid_xyz'][2]}_pre{pre_id}_post{post_id}"
                ),
            )
        )

    if records:
        store.save_records(records)


def make_queueable_tasks(
    payloads: Iterable[dict],
    *,
    store_backend: str,
    store_kwargs: dict | None = None,
) -> Iterable[partial]:
    """Convert JSON payloads into queueable partials ready for insertion."""

    for payload in payloads:
        yield partial(
            process_task,
            payload=payload,
            store_backend=store_backend,
            store_kwargs=store_kwargs or {},
        )
