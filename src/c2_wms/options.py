"""Sampler options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplerOptions:
    """Configuration that affects sampler compilation and output."""

    seed: int | None = None
    validate_masses: bool = True
    max_trace_states: int | None = None
