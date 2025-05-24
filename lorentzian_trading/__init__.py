"""
Lorentzian Trading Core Library

A quantum-inspired machine learning library for financial trading that uses
Lorentzian distance metrics to account for the warping effects of significant
economic events on price-time relationships.
"""

__version__ = "0.1.0"
__author__ = "Yassir DANGOU"
__email__ = "dangouyassir3@gmail.com"

from .core.classifier import LorentzianClassifier
from .core.distance import LorentzianDistance
from .features.engineering import FeatureEngine
from .filters.market import MarketFilters
from .backtesting.engine import BacktestEngine
from .utils.data import DataProcessor
from .kernels.regression import KernelRegression

__all__ = [
    'LorentzianClassifier',
    'LorentzianDistance',
    'FeatureEngine',
    'MarketFilters',
    'BacktestEngine',
    'DataProcessor',
    'KernelRegression',
]
