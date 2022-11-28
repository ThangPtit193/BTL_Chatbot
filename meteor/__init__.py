try:
    from importlib import metadata
except (ModuleNotFoundError, ImportError):
    # Python <= 3.7
    import importlib_metadata as metadata

try:
    __version__: str = str(metadata.version("meteor"))
except:
    __version__ = "0.0.1"

from meteor.schema import Document, Answer, Label, MultiLabel, Span, EvaluationResult
from meteor.nodes.base import BaseComponent
from meteor.pipelines.base import Pipeline
