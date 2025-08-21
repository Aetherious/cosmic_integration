"""Deprecated I/O helpers for binned analyses.

The original routines provided helpers for recursively loading nested
HDF5 groups into dictionaries.  In the dasein rewrite these utilities
are not required.  A placeholder is provided to satisfy imports.
"""

from __future__ import annotations

from typing import Any, Dict


def recursively_load_dict_contents_from_group(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Placeholder for recursive HDF5 group loading.

    Raises
    ------
    NotImplementedError
        Always raised as this function is not available in the dasein
        rewrite.
    """
    raise NotImplementedError(
        "recursively_load_dict_contents_from_group is not available in the dasein rewrite."
    )