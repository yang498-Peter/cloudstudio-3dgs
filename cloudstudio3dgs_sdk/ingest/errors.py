"""Fail-closed error types for the ingestion layer.

Every refusal names the file or key that was missing.  A generic
``KeyError``/``FileNotFoundError`` escaping the adapters is a bug: the caller
is a pipeline runner that cannot guess which of a dozen inputs was absent.
"""

from __future__ import annotations


class IngestError(Exception):
    """Base class for every ingestion refusal."""


class DatasetDetectionError(IngestError):
    """No adapter claimed the path, or more than one did."""


class DatasetIncompleteError(IngestError):
    """An adapter claimed the path but a required input is missing or malformed."""


class BundleSignatureError(IngestError):
    """A bundle manifest is unsigned, altered, or bound to different inputs."""


class GpuStepRequired(IngestError):
    """A cache in the plan needs CUDA and must not be started from here."""


__all__ = [
    "BundleSignatureError",
    "DatasetDetectionError",
    "DatasetIncompleteError",
    "GpuStepRequired",
    "IngestError",
]
