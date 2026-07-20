from importlib import import_module
from typing import Any

__all__ = [
    "app",
    "process_queue_job",
    "return_ctc_edges",
    "return_seg_edge",
    "return_volume_counts",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".cloudome", __name__)
    return getattr(module, name)
