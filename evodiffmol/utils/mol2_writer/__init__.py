"""
MOL2 Writer Module

This module provides MOL2 format writing capabilities for molecules.
MOL2 is the most reliable format for complex molecules with aromatic systems.

Main function:
- write_mol2_structure: Write 3D structure in MOL2 format
"""

from .mol2_writer import write_mol2_structure

__all__ = [
    'write_mol2_structure'
] 