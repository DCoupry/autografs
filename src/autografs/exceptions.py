"""Typed exceptions for AuToGraFS.

Using dedicated exception classes lets callers distinguish expected
generation failures (an SBU that does not fit a slot) from genuine
bugs, instead of relying on asserts or blanket exception handlers.
"""

from __future__ import annotations


class AutografsError(Exception):
    """Base class for all AuToGraFS errors."""


class AlignmentError(AutografsError):
    """Raised when aligning SBUs onto topology slots exceeds tolerance."""


class TopologyExtractionError(AutografsError):
    """Raised when a net cannot be converted into an AuToGraFS topology."""
