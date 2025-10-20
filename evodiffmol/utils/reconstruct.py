import torch
from rdkit import Chem
from rdkit import Geometry
import numpy as np

atom_decoder = ['H', 'C', 'N', 'O', 'F']
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H': 119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243},
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}

bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

margin1, margin2, margin3 = 5, 2, 1


# margin1, margin2, margin3 = 5, 3, 1
# margin1, margin2, margin3 = 10, 5, 1
# margin1, margin2, margin3 = 0, 0, 0

def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """ p: atom pair (couple of str)
        l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order


def num_confs(num: str):
    if num.endswith('x'):
        return lambda x: x * int(num[:-1])
    elif int(num) > 0:
        return lambda x: int(num)
    else:
        raise ValueError()


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def mol2smiles(mol):
    """
    Convert RDKit molecule to canonical SMILES string.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        str: Canonical SMILES string, or None if conversion fails
    """
    if mol is None:
        return None
    
    try:
        # Simple conversion with sanitization
        Chem.SanitizeMol(mol)
        # Remove all radicals before generating canonical SMILES
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() != 0:
                atom.SetNumRadicalElectrons(0)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles
        
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        # If conversion fails, return None
        return None



def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    atom_decoder = dataset_info['atom_decoder']
    
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info[
                'name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom' or dataset_info['name'] == 'crossdock':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j],
                                       limit_bonds_to_one=False)
            else:
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E





def set_positions(rd_mol, pos):
    n_atoms = len(rd_mol.GetAtoms())
    rd_conf = Chem.Conformer(n_atoms)
    xyz = pos
    # add atoms and coordinates
    for i in range(n_atoms):
        rd_coords = Geometry.Point3D(*xyz[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    return rd_mol



def build_molecule_openbabel(positions, atom_types, dataset_info):
    """
    Build a molecule using Open Babel to assign bond orders and aromaticity from 3D coordinates and atom types.
    Returns an RDKit molecule. If Open Babel fails, raise an error instead of falling back.
    """
    from openbabel import openbabel
    import pybel
    from rdkit import Chem

    # Use original positions without fixing
    positions_fixed = positions
    
    atom_decoder = dataset_info['atom_decoder']
    obmol = openbabel.OBMol()
    atom_indices = []
    for i, at in enumerate(atom_types):
        symbol = atom_decoder[at]
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(openbabel.GetAtomicNum(symbol))
        x, y, z = positions_fixed[i]
        obatom.SetVector(float(x), float(y), float(z))
        atom_indices.append(obatom.GetIdx())
    
    # Connect atoms and perceive bond orders
    try:
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
    except Exception as e:
        print(f"Warning: OpenBabel bond perception failed ({e}), molecule is invalid")
        return None
    
    # Convert to RDKit molecule using OBConversion
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat("mol")
    molblock = obConversion.WriteString(obmol)
    
    # Check if molblock generation succeeded
    if not molblock or len(molblock.strip()) == 0:
        print("Warning: OpenBabel failed to generate MOL block, molecule is invalid")
        return None
    
    # Disable sanitization to preserve all atoms from OpenBabel
    mol = Chem.MolFromMolBlock(molblock, sanitize=False)
    
    # Check if RDKit could parse the molblock
    if mol is None:
        print("Warning: RDKit failed to parse OpenBabel MOL block, molecule is invalid")
        return None
    
    # Try to sanitize - if it fails, the molecule is chemically invalid
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Warning: Sanitization failed ({e}), molecule is chemically invalid")
        return None  # Return None for invalid molecules

    # if mol is None:
    #     raise RuntimeError("Open Babel failed to build molecule from 3D coordinates. No fallback allowed.")
    
    # # Try to sanitize the molecule
    # try:
    #     Chem.SanitizeMol(mol)
    #     return mol
    # except Exception as e:
    #     raise RuntimeError(f"Open Babel built molecule, but RDKit sanitization failed: {e}")


def build_molecule_basic(positions, atom_types, dataset_info):
    """
    Basic molecule building using XAE (atom types, adjacency matrix, edge types) approach.
    This is the original MDM implementation for distance-based bond assignment.
    """
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(dataset_info['atom_decoder'][atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol


# Flexible molecule building with method selection
def build_molecule(positions, atom_types, dataset_info, method='openbabel'):
    """
    Build a molecule from 3D coordinates and atom types using specified method.
    
    Args:
        positions: 3D coordinates tensor/array
        atom_types: Atom type indices tensor/array
        dataset_info: Dataset configuration dictionary
        method: Method to use for bond perception ('openbabel' or 'basic')
                - 'openbabel': Use OpenBabel for sophisticated bond perception (default)
                - 'basic': Use basic distance-based bond assignment
    
    Returns:
        RDKit molecule object or None if failed
    """
    if method == 'openbabel':
        try:
            return build_molecule_openbabel(positions, atom_types, dataset_info)
        except Exception as e:
            print(f"Warning: OpenBabel molecule building failed: {e}")
            return None
    elif method == 'basic':
        try:
            return build_molecule_basic(positions, atom_types, dataset_info)
        except Exception as e:
            print(f"Warning: Basic molecule building failed: {e}")
            return None
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'openbabel' or 'basic'.")
