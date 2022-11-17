from __future__ import annotations

from typing import Callable

registered_metrics = {}
registered_reports = {}


def register_metrics(name: str, implementation: Callable):  # pylint: disable=missing-function-docstring
    registered_metrics[name] = implementation


def register_report(name: str, implementation: Callable):
    """
    Register a custom reporting function to be used during eval.

    This can be useful:
    - if you want to overwrite a report for an existing output type of prediction head (e.g. "per_token")
    - if you have a new type of prediction head and want to add a custom report for it

    :param name: This must match the `ph_output_type` attribute of the PredictionHead for which the report should be used.
                 (e.g. TokenPredictionHead => `per_token`, YourCustomHead => `some_new_type`).
    :param implementation: Function to be executed. It must take lists of `y_true` and `y_pred` as input and return a
                           printable object (e.g. string or dict).
                           See sklearns.metrics.classification_report for an example.
    :type implementation: function
    """
    registered_reports[name] = implementation


def _relevance_scores(y_true, y_pred):
    pass
