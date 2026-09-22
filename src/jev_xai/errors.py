"""Exception hierarchy for jev-xai."""

from __future__ import annotations


class JevXaiError(Exception):
    """Base error for every failure raised by jev-xai."""


class JevXaiUsageError(JevXaiError):
    """The caller used the library in a way that cannot work.

    Raised, for example, when a sync method is called from inside a running
    event loop. Await the async method instead.
    """


class CapabilityError(JevXaiError):
    """A prerequisite for the requested explainer is missing."""

    def __init__(self, prerequisite: str, fix: str) -> None:
        self.prerequisite = prerequisite
        self.fix = fix
        super().__init__(f"missing prerequisite '{prerequisite}': {fix}")


class ModelCallError(JevXaiError):
    """The decision model raised, timed out, or returned an unusable value."""


class ReplayMismatchError(JevXaiError):
    """Behavioral reproduction failed, or evidence replay has no cassette entry."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        self.reason = reason
        super().__init__(message)


class SchemaVersionError(JevXaiError):
    """The record schema version is not readable by this release."""


class BudgetExceededError(JevXaiError):
    """A configured model-call budget was exhausted."""
