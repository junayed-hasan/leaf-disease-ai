#!/usr/bin/env python3
"""
TomatoLeaf-AI: Cross-Domain Tomato Leaf Disease Detection
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

requirements = read_requirements('requirements.txt')
requirements_balancing = read_requirements('requirements_balancing.txt')

setup(
    name="tomatoleaf-ai",
    version="1.0.0",
    author="Mohammad Junayed Hasan",
    author_email="junayed.hasan@example.com",
    description="Cross-Domain Tomato Leaf Disease Detection with Unified Optimization Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/junayed-hasan/lightweight-tomato",
    project_urls={
        "Bug Tracker": "https://github.com/junayed-hasan/lightweight-tomato/issues",
        "Documentation": "https://github.com/junayed-hasan/lightweight-tomato/docs",
        "Source Code": "https://github.com/junayed-hasan/lightweight-tomato",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "balancing": requirements_balancing,
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "tomatoleaf-train=scripts.train:main",
            "tomatoleaf-evaluate=src.evaluation.evaluate_kd_on_test_datasets:main",
            "tomatoleaf-quantize=src.quantization.mobile_quantization_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    keywords=[
        "deep learning",
        "computer vision",
        "plant disease detection",
        "knowledge distillation",
        "ensemble learning",
        "quantization",
        "mobile deployment",
        "agriculture",
        "tomato",
        "cross-domain",
    ],
    zip_safe=False,
) 