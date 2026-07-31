"""Backward-compatible wrapper around the v9 chemical resolver."""
from .resolver import resolve_ligand


def resolve_name(query, timeout=8):
    result = resolve_ligand(str(query), int(timeout))
    if not result.get("success"):
        raise LookupError(result.get("message", "Ligand resolution failed."))
    return result
