"""Public error hierarchy for c2-wms."""


class SamplingError(RuntimeError):
    """Base class for sampling failures."""


class UnsupportedSamplingInput(SamplingError):
    """The WFOMC input is countable but does not define a sampling measure."""


class UnsatisfiableProblemError(SamplingError):
    """The problem has zero total weight."""


class WfomcCompatibilityError(SamplingError):
    """The pinned WFOMC internal contract differs from the expected shape."""
