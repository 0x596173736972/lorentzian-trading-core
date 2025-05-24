
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lorentzian-trading-core",
    version="0.1.0",
    author="Yassir DANGOU",
    author_email="dangouyassir3@gmail.com",
    description="A quantum-inspired machine learning library for financial trading using Lorentzian distance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0x596173736972/lorentzian-trading-core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8", "mypy>=0.812"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=0.5"],
        "plotting": ["plotly>=5.0", "matplotlib>=3.0"],
    },
    entry_points={
        "console_scripts": [
            "lorentzian-backtest=lorentzian_trading.cli:main",
        ],
    },
)
