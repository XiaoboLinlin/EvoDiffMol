"""
Setup script for EvoDiffMol package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f 
                if line.strip() and not line.startswith('#')]

# Read the long description from README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "EvoDiffMol: Molecular Generation and Optimization using Diffusion Models and Genetic Algorithms"

setup(
    name="evodiffmol",
    version="1.0.0",
    author="EvoDiffMol Team",
    description="Molecular generation and optimization using genetic algorithms and diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/EvoDiffMol",
    packages=find_packages(include=['evodiffmol', 'evodiffmol.*']),
    python_requires='>=3.8',
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='molecular-generation drug-discovery diffusion-models genetic-algorithms',
)
