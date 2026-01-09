"""Compatibility shim.

`intensity_mapper.py` used to host CT→mesh mapping utilities.
The source of truth is now `morpho_mapper.py` (richer: warps, cohort stats, etc.).

Keep this module to avoid breaking older analysis scripts.
"""

from morpho_mapper import export_vtx, idw, rescale_to_density

__all__ = [
    "idw",
    "rescale_to_density",
    "export_vtx",
]
