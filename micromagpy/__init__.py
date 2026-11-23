"""A small finite-difference micromagnetic simulator built on NumPy, SciPy and JAX."""

from importlib.metadata import PackageNotFoundError, version

try:  # the version is derived from git tags at build time (hatch-vcs)
    __version__ = version("micromagpy")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without installing
    __version__ = "0.0.0+unknown"
