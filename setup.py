"""
setup.py — allows 'pip install -e .' during local development
and installation via Databricks Repos or init scripts.
"""

from setuptools import setup, find_packages

setup(
    name="gov_finance_ml",
    version="1.0.0",
    description="State-level Government Finance Analytics & ML Pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where=".", include=["src*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "prophet>=1.1",
        "mlflow>=2.10",
        "pandas>=2.0",
        "numpy>=1.26",
        "pyarrow>=14.0",
        "pyyaml>=6.0",
        "delta-spark>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "ruff>=0.4",
            "mypy>=1.8",
            "matplotlib>=3.8",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "gov-finance-ingest=src.data.ingestion:main",
        ]
    },
)
