"""Custom exception types used across the app.

Keeping these small and specific lets callers (UI, CLI) decide how much detail
to surface to the end user without parsing generic exception messages.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for errors that carry a user-safe message."""


class ConfigError(AppError):
    """Raised when required configuration (e.g. an API key) is missing or invalid."""


class PipelineError(AppError):
    """Raised when a pipeline step fails in a way that should stop the run."""
