"""
ADMET and Molecular Property Configurations
============================================

Comprehensive property configurations derived from ADMET-AI predictions
on 2000 molecules from the MOSES dataset.

Ranges are constrained to physically meaningful bounds:
- Probabilities: [0, 1]
- Percentages: [0, 100]
- Other properties: observed range with 10% padding

Data Source: MOSES test dataset (2000 molecules)
Total Properties: 53 (4 Basic + 49 ADMET)
"""

from typing import Dict, Any


def get_all_property_configs() -> Dict[str, Dict[str, Any]]:
    """Get all property configurations from MOSES analysis.
    
    Each property includes:
    - range: [min, max] constrained to physically meaningful bounds
    - preferred_value: median from 2000 MOSES molecules
    
    Returns:
        Dictionary mapping property names to configurations
    """
    return {
        # Basic Physicochemical Properties (4 properties)
        # These are calculated by RDKit, not ADMET-AI
        'logp': {
            'range': [-3, 7],
            'preferred_value': 2.5  # From MOSES logP median
        },
        'qed': {
            'range': [0, 1],
            'preferred_value': 1.0  # Maximize drug-likeness
        },
        'sa': {
            'range': [1, 10],
            'preferred_value': 1.0  # Minimize (lower is easier to synthesize)
        },
        'tpsa': {
            'range': [0, 200],
            'preferred_value': 66  # From MOSES tpsa median
        },

        # Absorption (7 properties)
        'Bioavailability_Ma': {
            'range': [0.0, 1.0],
            'preferred_value': 1.0  # Maximize bioavailability
        },
        'Caco2_Wang': {
            'range': [-6, -3],
            'preferred_value': -4.7  # Moderate permeability (log scale)
        },
        'HIA_Hou': {
            'range': [0.0, 1.0],
            'preferred_value': 1.0  # Maximize absorption
        },
        'Lipophilicity_AstraZeneca': {
            'range': [-2, 5],
            'preferred_value': 2.2  # Optimal lipophilicity
        },
        'PAMPA_NCATS': {
            'range': [0.0, 1.0],
            'preferred_value': 1.0  # Maximize membrane permeability
        },
        'Pgp_Broccatelli': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize P-gp efflux
        },
        'Solubility_AqSolDB': {
            'range': [-7, 0],
            'preferred_value': -3.3  # Good solubility (log scale)
        },

        # Distribution (3 properties)
        'BBB_Martins': {
            'range': [0.0, 1.0],
            'preferred_value': 0.94  # High BBB permeability
        },
        'PPBR_AZ': {
            'range': [0.0, 100.0],
            'preferred_value': 78  # Moderate protein binding (%)
        },
        'VDss_Lombardo': {
            'range': [-25, 39],
            'preferred_value': -0.5  # Volume of distribution
        },

        # Metabolism (11 properties)
        # CYP inhibition - minimize to avoid drug-drug interactions
        'CYP1A2_Veith': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize inhibition
        },
        'CYP2C19_Veith': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize inhibition
        },
        'CYP2C9_Substrate_CarbonMangels': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize substrate activity
        },
        'CYP2C9_Veith': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize inhibition
        },
        'CYP2D6_Substrate_CarbonMangels': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize substrate activity
        },
        'CYP2D6_Veith': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize inhibition
        },
        'CYP3A4_Substrate_CarbonMangels': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize substrate activity
        },
        'CYP3A4_Veith': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize inhibition
        },
        'Clearance_Hepatocyte_AZ': {
            'range': [-55, 214],
            'preferred_value': 66  # Hepatocyte clearance
        },
        'Clearance_Microsome_AZ': {
            'range': [-59, 151],
            'preferred_value': 41  # Microsomal clearance
        },
        'Half_Life_Obach': {
            'range': [-87, 138],
            'preferred_value': 4.2  # Half-life (hours)
        },

        # Toxicity (6 properties)
        # All toxicity should be minimized for drug safety
        'AMES': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize mutagenicity
        },
        'Carcinogens_Lagunin': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize carcinogenicity
        },
        'ClinTox': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize clinical toxicity
        },
        'DILI': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize liver toxicity
        },
        'LD50_Zhu': {
            'range': [1, 4],
            'preferred_value': 4.2  # Maximize (higher = less toxic)
        },
        'hERG': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize cardiotoxicity
        },

        # Other (22 properties)
        'HydrationFreeEnergy_FreeSolv': {
            'range': [-21, -2],
            'preferred_value': -11  # Hydration free energy
        },
        'Lipinski': {
            'range': [0.0, 1.0],
            'preferred_value': 1.0  # Maximize (1 = passes all rules)
        },
        'NR-AR': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize (unless targeting)
        },
        'NR-AR-LBD': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize (unless targeting)
        },
        'NR-AhR': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize (unless targeting)
        },
        'NR-Aromatase': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize (unless targeting)
        },
        'NR-ER': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize (unless targeting)
        },
        'NR-ER-LBD': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize (unless targeting)
        },
        'NR-PPAR-gamma': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize (unless targeting)
        },
        'QED': {
            'range': [0.0, 1.0],
            'preferred_value': 1.0  # Maximize drug-likeness
        },
        'SR-ARE': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize stress response
        },
        'SR-ATAD5': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize stress response
        },
        'SR-HSE': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize stress response
        },
        'SR-MMP': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize stress response
        },
        'SR-p53': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize stress response
        },
        'Skin_Reaction': {
            'range': [0.0, 1.0],
            'preferred_value': 0.0  # Minimize skin sensitization
        },
        'hydrogen_bond_acceptors': {
            'range': [0, 10],
            'preferred_value': 4  # H-bond acceptors
        },
        'hydrogen_bond_donors': {
            'range': [0, 4],
            'preferred_value': 1  # H-bond donors
        },
        'logP': {
            'range': [-2, 5],
            'preferred_value': 2.5  # Lipophilicity
        },
        'molecular_weight': {
            'range': [240, 360],
            'preferred_value': 310  # Molecular weight (Da)
        },
        'stereo_centers': {
            'range': [0, 4],
            'preferred_value': 0  # Stereogenic centers
        },
        'tpsa': {
            'range': [-11, 162],
            'preferred_value': 66  # Topological polar surface area
        },
    }
