"""SCROLLS task normalization, prompting, batching, and sampling."""

from .scrolls import NormalizedExample, normalize_scrolls_example
from .tasks import TASKS, TaskSpec

__all__ = ["NormalizedExample", "TASKS", "TaskSpec", "normalize_scrolls_example"]
